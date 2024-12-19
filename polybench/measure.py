#!/usr/bin/env python3

import argparse
import subprocess
import csv
import sys
import json
from saleae import automation


def main():
    flash_command = sys.stdin.read().strip()
    export_filepath = "/Users/byeongje/wasm/mlir-wasm-dialect/polybench/result"

    # Connect to the running Logic 2 Application
    with automation.Manager.connect(port=10430) as manager:
        # Configure the device
        device_configuration = automation.LogicDeviceConfiguration(
            enabled_digital_channels=[0],  # Assuming signal is on digital channel 0
            digital_sample_rate=10_000_000,  # 10 MSa/s
        )

        # Configure the capture with a digital trigger on falling edge
        capture_configuration = automation.CaptureConfiguration(
            capture_mode=automation.DigitalTriggerCaptureMode(
                trigger_type=automation.DigitalTriggerType.FALLING,  # Trigger on falling edge
                trigger_channel_index=0,  # Monitor channel 0 for the trigger
                after_trigger_seconds=0.001,  # Capture an additional 1ms after the trigger
            )
        )

        # Start the capture
        with manager.start_capture(
            device_configuration=device_configuration,
            capture_configuration=capture_configuration,
        ) as capture:
            # Execute the flash command
            subprocess.Popen(
                flash_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()

            # Wait for the flash process to complete
            # Wait for the capture to complete automatically after the falling edge
            capture.wait()
            # print("Capture completed automatically after falling edge.")

            # Export the captured data to a CSV file
            capture.export_raw_data_csv(export_filepath)
            # print(f"Data exported to {export_filepath}")

    # Analyze the exported data to measure the time window
    rising_edge_time = None
    falling_edge_time = None

    exported_file = f"{export_filepath}/digital.csv"

    with open(exported_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            timestamp = float(row[0])  # Time in seconds
            state = int(row[1])  # Pin state (0 or 1)

            # Detect rising edge
            if state == 1 and rising_edge_time is None:
                rising_edge_time = timestamp

            # Detect falling edge after the rising edge
            if state == 0 and rising_edge_time is not None:
                falling_edge_time = timestamp
                break

    # Calculate and store the time window
    if rising_edge_time is not None and falling_edge_time is not None:
        time_window = (
            falling_edge_time - rising_edge_time
        ) * 1_000_000.0  # Convert to microseconds
        print(f"[execution time] {time_window:.3f} microseconds")
    else:
        print("Failed to detect the complete pulse sequence.")


if __name__ == "__main__":
    main()
