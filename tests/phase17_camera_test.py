#!/usr/bin/env python3
"""Phase 17 acceptance test: the camera as live video for the app (step 7 of the Android app).

Needs two things running first:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts vacuum_video)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase17_camera_test.py

A scripted viewer does what the app does: posts a WebRTC offer to http://localhost:8080/offer
and receives the video.
  1. nobody watching    vacuum_video is not subscribed to the camera: no copying, no encoding
  2. pictures           over 20 s the viewer receives at least 90% of the pictures the camera
                        publishes, at the camera's size. (The design asked for 12 a second, for a
                        15-a-second camera. The simulator renders its camera at about 10 a
                        second of simulated time and runs at about 0.8 of real time, so it
                        publishes 6 to 9 a second: the video can only pass on what it gets.)
  3. delay              how long a picture takes from the camera topic to the viewer, decoded.
                        Measured while the robot turns, by matching each received picture to the
                        camera's pictures. Target: under 0.5 s
  4. viewer leaves      once the viewer has gone, vacuum_video unsubscribes within 5 s
"""

import asyncio
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

import numpy as np
import rclpy
from aiortc import RTCPeerConnection, RTCSessionDescription
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from mode_client import ModeClient  # noqa: E402
from phase7_mode_manager_test import ModeTestDriver  # noqa: E402

OFFER_URL = "http://127.0.0.1:8080/offer"
CAMERA = "/camera/image_raw"
WATCH_SECONDS = 20.0


def signature(gray: np.ndarray) -> np.ndarray:
    """A picture boiled down to 16 x 12 averages, normalised, so that a picture and its video
    copy (compressed, perhaps in a slightly different brightness range) still match."""
    h, w = gray.shape
    small = gray[: h - h % 12, : w - w % 16].reshape(12, h // 12, 16, w // 16).mean(axis=(1, 3)).ravel()
    small = small - small.mean()
    norm = np.linalg.norm(small)
    return small / norm if norm > 0 else small


class Viewer:
    """A scripted app: its own thread and asyncio loop, recording (arrival time, width, height,
    signature) of every picture."""

    def __init__(self):
        self.pictures = []
        self.error = None
        self.connected_at = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(10.0)

    async def _run(self):
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")

        @pc.on("track")
        def on_track(track):
            async def read():
                while True:
                    try:
                        frame = await track.recv()
                    except Exception:
                        return
                    gray = frame.to_ndarray(format="gray")
                    self.pictures.append((time.monotonic(), frame.width, frame.height, signature(gray.astype(np.float32))))

            asyncio.ensure_future(read())

        @pc.on("connectionstatechange")
        def on_state():
            if pc.connectionState == "connected" and self.connected_at is None:
                self.connected_at = time.monotonic()

        try:
            await pc.setLocalDescription(await pc.createOffer())
            body = json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}).encode()
            request = urllib.request.Request(OFFER_URL, data=body, headers={"Content-Type": "application/json"})
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, lambda: json.loads(urllib.request.urlopen(request, timeout=10).read()))
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        except Exception as exc:
            self.error = repr(exc)
        finally:
            await pc.close()


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    results = []
    camera = []  # (arrival, signature) of every picture on the camera topic

    def on_image(msg):
        rgb = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        rgb = rgb.reshape(msg.height, msg.width, 3).astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        camera.append((time.monotonic(), signature(gray)))

    # Reliable, as the video node reads it: best-effort lost about 40 % of these 1 MB pictures, so
    # this count of what the camera published was too low.
    n.create_subscription(Image, CAMERA, on_image, rclpy.qos.QoSProfile(depth=5, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE))

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def video_subscribed():
        return any(info.node_name == "vacuum_video" for info in n.get_subscriptions_info_by_topic(CAMERA))

    def wait_until(condition, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            n.spin(0.05)
            if condition():
                return True
        return condition()

    viewer = None
    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        if not wait_until(lambda: len(camera) > 10, 30.0):
            check("robot ready", False, f"no pictures on {CAMERA}")
            return results
        n.wait_sim(2.0)

        # 1. nobody watching
        node_seen = wait_until(lambda: "vacuum_video" in n.get_node_names(), 20.0)
        check("nobody watching", node_seen and not video_subscribed(),
              f"vacuum_video running: {node_seen}; subscribed to the camera with nobody watching: {video_subscribed()}")

        # 2 and 3. watch for 20 s while the robot turns
        modes.request("manual")
        viewer = Viewer().start()
        connected = wait_until(lambda: viewer.connected_at is not None and len(viewer.pictures) > 5, 20.0)
        subscribed_while_watching = video_subscribed()
        t0 = time.monotonic()
        n.hold(0.0, 0.6)  # turning, so that consecutive pictures differ and can be matched
        wait_until(lambda: time.monotonic() - t0 >= WATCH_SECONDS, WATCH_SECONDS + 5.0)
        n.release()
        modes.request("idle")
        got = [p for p in viewer.pictures if t0 <= p[0] <= t0 + WATCH_SECONDS]
        shown = [p for p in camera if t0 <= p[0] <= t0 + WATCH_SECONDS]
        rate = len(got) / WATCH_SECONDS
        sizes = sorted({(w, h) for _, w, h, _ in got})
        share = len(got) / len(shown) if shown else 0.0
        check("pictures", connected and shown and share >= 0.9 and sizes == [(640, 480)] and subscribed_while_watching,
              f"connected: {connected} ({viewer.error or 'no error'}); {len(got)} pictures in {WATCH_SECONDS:.0f} s = "
              f"{rate:.1f} a second, {share:.0%} of the {len(shown)} the camera published ({len(shown) / WATCH_SECONDS:.1f} "
              f"a second); sizes {sizes}; subscribed while watching: {subscribed_while_watching}")

        delays = []
        for arrived, _, _, sig in got:
            recent = [(t, s) for t, s in camera if arrived - 2.0 <= t <= arrived]
            if len(recent) < 3:
                continue
            scores = [float(np.dot(sig, s)) for _, s in recent]
            best = int(np.argmax(scores))
            if scores[best] > 0.95:
                delays.append(arrived - recent[best][0])
        median = statistics.median(delays) if delays else None
        worst = max(delays) if delays else None
        check("delay", median is not None and median < 0.5,
              f"{len(delays)} of {len(got)} pictures matched to the camera's; delay median "
              f"{median * 1000 if median is not None else float('nan'):.0f} ms, worst "
              f"{worst * 1000 if worst is not None else float('nan'):.0f} ms (target: under 500 ms)")

        # 4. the viewer leaves
        viewer.stop()
        viewer = None
        t4 = time.monotonic()
        gone = wait_until(lambda: not video_subscribed(), 5.0)
        check("viewer leaves", gone, f"unsubscribed from the camera {round(time.monotonic() - t4, 2) if gone else 'not'} "
              f"within 5 s of the viewer closing")

        out = os.environ.get("PHASE17_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results}, f)
    finally:
        if viewer is not None:
            viewer.stop()
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
