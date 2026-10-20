#!/usr/bin/env python3
"""Exercise bag synchronization with tiny synthetic bags, without dataset files."""

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("Install examples/requirements.txt to run the bag stream tests.")
    raise SystemExit(77)


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "examples/new_college_bag_stream.py"
TOPICS = [f"/alphasense_driver_ros/cam{camera}/compressed" for camera in (0, 1, 3, 4)]
TYPESTORE = get_typestore(Stores.ROS1_NOETIC)
IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
TIME = TYPESTORE.types["builtin_interfaces/msg/Time"]
HEADER = TYPESTORE.types["std_msgs/msg/Header"]
IMAGE = TYPESTORE.types[IMAGE_TYPE]


class BagStreamCheck(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bag = Path(self.directory.name) / "input.bag"

    def write_bag(self, events, cameras=4, reconnect=False):
        """Events are (camera index, header timestamp, payload marker, restarted)."""
        with Writer(self.bag) as writer:
            connections = [
                writer.add_connection(topic, IMAGE_TYPE, typestore=TYPESTORE)
                for topic in TOPICS[:cameras]
            ]
            restarted = None
            if reconnect:
                restarted = writer.add_connection(
                    TOPICS[0], IMAGE_TYPE, typestore=TYPESTORE, callerid="/restarted"
                )
            for record_index, (camera, timestamp, marker, use_restart) in enumerate(events):
                message = IMAGE(
                    HEADER(record_index, TIME(*divmod(timestamp, 1_000_000_000)), ""),
                    "mono8; jpeg compressed",
                    np.array([marker, camera], dtype=np.uint8),
                )
                connection = restarted if use_restart else connections[camera]
                # Recorder timestamps intentionally differ from image header timestamps.
                writer.write(
                    connection,
                    10_000_000_000 + record_index,
                    TYPESTORE.serialize_ros1(message, IMAGE_TYPE),
                )

    def run_helper(self, *options):
        command = [sys.executable, str(HELPER), "--bag", str(self.bag)]
        for topic in TOPICS:
            command += ["--topic", topic]
        return subprocess.run(command + list(options), capture_output=True, timeout=20)

    def read_frames(self, data):
        self.assertEqual(data[:8], b"XPLNCB01")
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], 4)
        offset = 12
        frames = []
        while offset < len(data):
            timestamp = struct.unpack_from("<q", data, offset)[0]
            offset += 8
            images = []
            for _ in range(4):
                size = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                images.append(data[offset : offset + size])
                offset += size
            frames.append((timestamp, images))
        self.assertEqual(offset, len(data))
        return frames

    def test_sync_sampling_topic_order_and_reconnection(self):
        events = []
        for frame in range(5):
            timestamp = 1_000_000_000 + frame * 1_000_000
            if frame == 1:
                events.append((0, timestamp - 500_000, 99, False))
            for camera in (3, 1, 0, 2):
                events.append(
                    (camera, timestamp + camera * 100, frame, camera == 0 and frame >= 2)
                )
        self.write_bag(events, reconnect=True)
        result = self.run_helper("--start", "1", "--step", "2", "--max-frames", "2")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        frames = self.read_frames(result.stdout)
        self.assertEqual([f[0] for f in frames], [1_001_000_000, 1_003_000_000])
        for marker, (_, images) in zip((1, 3), frames):
            self.assertEqual(images, [bytes([marker, camera]) for camera in range(4)])
        self.assertIn(b"discarded_images=1", result.stderr)

    def test_missing_topic_fails_before_stream_header(self):
        self.write_bag([(i, 1_000_000_000, 0, False) for i in range(3)], cameras=3)
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"missing image topics", result.stderr)
        self.assertEqual(result.stdout, b"")

    def test_nonincreasing_image_timestamp_fails(self):
        self.write_bag(
            [(i, 1_000_000_000, 0, False) for i in range(4)]
            + [(0, 999_999_999, 1, False)]
        )
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"non-increasing image timestamp", result.stderr)

    def test_empty_selection_is_an_error(self):
        self.write_bag([(i, 1_000_000_000, 0, False) for i in range(4)])
        result = self.run_helper("--start", "10")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"no synchronized frames selected", result.stderr)
        self.assertEqual(self.read_frames(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
