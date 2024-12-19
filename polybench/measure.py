import subprocess
import csv
from saleae import automation

# Define your custom flash command
flash_command = "cd .. && ./run-mcu.sh polybench/atax_256.mlir --compiler=mlir --testcase=ATAX_MLIR --use-aot=true -- --opt-level=3 --target=thumbv7em --target-abi=eabihf --cpu=cortex-m4"  # Replace with your actual command

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
        print("Capture started...")

        # Execute the flash command
        process = subprocess.Popen(flash_command, shell=True)

        # Wait for the flash process to complete
        process.wait()
        print("Flash command completed!")

        # Wait for the capture to complete automatically after the falling edge
        capture.wait()
        print("Capture completed automatically after falling edge.")

        # Export the captured data to a CSV file
        capture.export_raw_data_csv(export_filepath)
        print(f"Data exported to {export_filepath}")

# Analyze the exported data to measure the time window
rising_edge_time = None
falling_edge_time = None

exported_file = export_filepath + "/digital.csv"

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

# Calculate and print the time window
if rising_edge_time is not None and falling_edge_time is not None:
    time_window = (falling_edge_time - rising_edge_time) * 1000.0
    print(f"Time between rising and falling edges: {time_window:.9f} microseconds")
else:
    print("Failed to detect the complete pulse sequence.")
