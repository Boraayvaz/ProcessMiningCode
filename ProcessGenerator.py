import random
import time
import threading
import openpyxl
import datetime

# Global variables for tracking the current event and cycle
current_event = 0
current_cycle = 0
data_to_write = []  # List to accumulate data for writing to Excel
write_lock = threading.Lock()  # Lock to ensure thread-safe writing

# Create the Excel workbook and worksheet
headers = ["Timestamp", "EventID", "ProcessID"]

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputfile = "Process"
outputfile_excel = f"{outputfile}_{current_datetime}.xlsx"
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "EventData"

# Write the headers
for col_num, header in enumerate(headers, 1):
    ws.cell(row=1, column=col_num, value=header)

# Function to periodically print the current event and cycle number
def print_status():
    global data_to_write
    while True:
        if current_event > 0 and current_cycle > 0:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with write_lock:
                data_to_write.append([timestamp, current_event - 1, current_cycle])
        time.sleep(0.1)

# Function to periodically write data to the Excel file
def write_to_excel():
    global data_to_write
    while True:
        with write_lock:
            if data_to_write:
                for row in data_to_write:
                    ws.append(row)
                wb.save(outputfile_excel)
                data_to_write = []  # Clear the list after writing
        time.sleep(5)  # Adjust the interval as needed, e.g., every 5 seconds

# Step 2: Function to get the number of events from the user
def get_number_of_events():
    while True:
        try:
            num_events = int(input("Enter the number of events: "))
            if num_events > 0:
                return num_events
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Step 3: Function to get the number of cycles from the user
def get_number_of_cycles():
    while True:
        try:
            num_cycles = int(input("Enter the number of cycles: "))
            if num_cycles > 0:
                return num_cycles
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Step 4: Function to get the interval for durations from the user
def get_duration_interval():
    while True:
        try:
            min_duration = float(input("Enter the minimum duration for an event (in seconds): "))
            max_duration = float(input("Enter the maximum duration for an event (in seconds): "))
            if min_duration > 0 and max_duration > min_duration:
                return min_duration, max_duration
            else:
                print("Please ensure the maximum duration is greater than the minimum and both are positive.")
        except ValueError:
            print("Invalid input. Please enter valid numbers.")

# Step 5: Function to get specific events for anomalies and their probabilities from the user
def get_anomaly_events_and_probabilities():
    while True:
        try:
            skip_events = list(map(int, input("Enter the event numbers to be skipped (comma-separated): ").split(',')))
            delay_events = list(map(int, input("Enter the event numbers to be delayed (comma-separated): ").split(',')))
            skip_prob = float(input("Enter the probability of skipping these events (0 to 1): "))
            delay_prob = float(input("Enter the probability of delaying these events (0 to 1): "))
            
            if all(0 <= event <= num_events for event in skip_events + delay_events) and 0 <= skip_prob <= 1 and 0 <= delay_prob <= 1:
                return skip_events, delay_events, skip_prob, delay_prob
            else:
                print("Please ensure all event numbers are within the valid range and probabilities are between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter valid numbers.")

# Step 6: Function to generate a flow of events with user-defined time intervals
def generate_event_flow(num_events, min_duration, max_duration):
    event_flow = []
    for i in range(num_events):
        event_name = str(i + 1)
        time_interval = random.uniform(min_duration, max_duration)  # Random interval based on user-defined range
        event_flow.append((event_name, time_interval))
    return event_flow

# Step 7: Function to introduce random anomalies based on user-defined events and probabilities
def introduce_anomalies(event_flow, skip_prob, delay_prob, skip_events, delay_events, min_duration, max_duration):
    anomalies = []
    for event in event_flow:
        event_number = int(event[0])  # Convert event name to integer for comparison
        if event_number in skip_events:
            if random.random() < skip_prob:
                anomalies.append((event[0], 0, 'skip'))  # Skip event
                continue
        if event_number in delay_events:
            if random.random() < delay_prob:
                new_time_interval = event[1] + random.uniform(min_duration*3, max_duration*3)  # Add extra time to the interval
                anomalies.append((event[0], new_time_interval, 'delay'))
                continue
        anomalies.append((event[0], event[1], 'none'))  # No anomaly
    return anomalies

# Step 8: Function to display the event flow with anomalies and simulate time intervals
def display_event_flow_with_anomalies(event_flow, num_cycles, skip_prob, delay_prob, skip_events, delay_events, min_duration, max_duration):
    global current_event, current_cycle
    for cycle in range(num_cycles):
        current_cycle = cycle + 1
        print(f"\nCycle {current_cycle}/{num_cycles}")
        anomalous_flow = introduce_anomalies(event_flow, skip_prob, delay_prob, skip_events, delay_events, min_duration, max_duration)
        for event_name, time_interval, anomaly in anomalous_flow:
            current_event = int(event_name)
            if anomaly == 'skip':
                print(f"Event: {event_name} (Skipped due to anomaly)")
                continue
            elif anomaly == 'delay':
                print(f"Event: {event_name} (Delayed, new wait time {time_interval:.2f} seconds)")
            else:
                print(f"Event: {event_name} (Wait for {time_interval:.2f} seconds)")
            time.sleep(time_interval)  # Simulate waiting time
        print(f"Cycle {current_cycle} completed.")

# Main application
def main():
    global num_events
    num_events = get_number_of_events()
    num_cycles = get_number_of_cycles()
    min_duration, max_duration = get_duration_interval()
    skip_events, delay_events, skip_prob, delay_prob = get_anomaly_events_and_probabilities()
    
    # Start the status printing thread
    status_thread = threading.Thread(target=print_status, daemon=True)
    status_thread.start()
    
    # Start the Excel writing thread
    excel_thread = threading.Thread(target=write_to_excel, daemon=True)
    excel_thread.start()
    
    # Generate the flow and simulate with anomalies
    event_flow = generate_event_flow(num_events, min_duration, max_duration)
    display_event_flow_with_anomalies(event_flow, num_cycles, skip_prob, delay_prob, skip_events, delay_events, min_duration, max_duration)

if __name__ == "__main__":
    main()
