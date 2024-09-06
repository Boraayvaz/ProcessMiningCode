import tkinter as tk
from tkinter import ttk

from tkinter import filedialog, messagebox
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors as rl_colors
from reportlab.platypus import Image

np.set_printoptions(precision=6, suppress=True)
# Hardcoded folder path
folder_path = r'C:\Users\BoraAyvaz\iCloudDrive\BeyzosThesis'

# Initialize variables to store selected headers
PMmsg = None
PMID = None

# Function to upload a file
def upload_file():
    global df  # Declare df as global to ensure it's accessible in other functions
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            df = pd.read_excel(file_path)
            headers = df.columns.tolist()
            
            # Clear the current listbox content
            headers_listbox.delete(0, tk.END)
            
            # Add headers to the listbox
            for header in headers:
                headers_listbox.insert(tk.END, header)
                
            label.config(text=f"File Uploaded: {file_path}")
        except Exception as e:
            label.config(text=f"Failed to upload file: {e}")

# Function to select the latest modified file from the hardcoded folder
def select_latest_file():
    try:
        file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~$')]
        if not file_list:
            messagebox.showinfo("No Files Found", f"No .xlsx files found in the folder: {folder_path}")
            return

        latest_file = ''
        latest_mod_date = datetime.min

        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            mod_date = datetime.fromtimestamp(os.path.getmtime(file_path))

            if mod_date > latest_mod_date:
                latest_mod_date = mod_date
                latest_file = file_name

        if not latest_file:
            messagebox.showinfo("No Valid Files", f"No valid .xlsx files found in the folder: {folder_path}")
        else:
            file_path = os.path.join(folder_path, latest_file)
            global df
            df = pd.read_excel(file_path)
            headers = df.columns.tolist()

            # Clear the current listbox content
            headers_listbox.delete(0, tk.END)

            # Add headers to the listbox
            for header in headers:
                headers_listbox.insert(tk.END, header)

            label.config(text=f"Latest File Selected: {latest_file}")
            
    except Exception as e:
        label.config(text=f"Failed to select file: {e}")

# Function to assign selected header to PMmsg
def assign_pmmsg():
    global df  # Ensure df is recognized as the global variable
    selected = headers_listbox.curselection()
    if len(selected) != 1:
        messagebox.showwarning("Selection Error", "Please select exactly one header for PMmsg.")
        return
    
    selected_column = headers_listbox.get(selected[0])
    
    # Check if the selected column exists in the DataFrame
    if selected_column not in df.columns:
        messagebox.showerror("Error", f"Column '{selected_column}' not found in the Excel file.")
        return
    
    global PMmsg
    PMmsg = df[selected_column].to_numpy()
    pmmsg_label.config(text=f"PMmsg: {selected_column}")
# Function to assign selected header to PMID
def assign_pmid():
    global df  # Ensure df is recognized as the global variable
    selected = headers_listbox.curselection()
    if len(selected) != 1:
        messagebox.showwarning("Selection Error", "Please select exactly one header for PMID.")
        return
    
    selected_column = headers_listbox.get(selected[0])
    
    # Check if the selected column exists in the DataFrame
    if selected_column not in df.columns:
        messagebox.showerror("Error", f"Column '{selected_column}' not found in the Excel file.")
        return
    
    global PMID
    PMID = df[selected_column].to_numpy()
    pmid_label.config(text=f"PMID: {selected_column}")

# Function to run the provided code after PMmsg and PMID are selected
def draw_curved_edges(G, pos, ax, edge_labels=None, edge_colors=None, linewidths=None, arrow_sizes=None):
    for (i, (u, v)) in enumerate(G.edges()):
        if edge_colors[i] == 1:
            ColorD = [0, 0, 1]
        else:
            ColorD = [1, 0.5, 1]

        
        rad = 0.1  # Controls the curvature; increase for more curve
        arrowprops = dict(arrowstyle="-|>", lw=linewidths[i], 
                          color=ColorD,  # Add color here
                          mutation_scale=arrow_sizes[i], shrinkA=10, shrinkB=10)
        edge_patch = FancyArrowPatch(posA=pos[u], posB=pos[v], connectionstyle=f"arc3,rad={rad}", **arrowprops)

        ax.add_patch(edge_patch)
        #print(edge_colors[i])
        # Adjust the position of edge labels for the curved edge
        if edge_labels:
            # Calculate a more appropriate midpoint for the curved edge
            rad_x = (pos[u][0] + pos[v][0]) / 2
            rad_y = (pos[u][1] + pos[v][1]) / 2
            # Offset the label a bit based on the curvature (rad) for better positioning
            label_offset = 0.1 * rad
            mid_x = rad_x + label_offset * (pos[v][1] - pos[u][1])
            mid_y = rad_y + label_offset * (pos[u][0] - pos[v][0])
            ax.text(mid_x, mid_y, edge_labels.get((u, v), ''), fontsize=6, color="black")


def run_analysis():
    if PMmsg is None or PMID is None:
        messagebox.showwarning("Missing Data", "Please assign both PMmsg and PMID before running the analysis.")
        return

    # Provided code to run after selections
    diff_PMIID = np.diff(PMID, prepend=0)
    NoP = np.count_nonzero(diff_PMIID)

    Process = []
    a = 0
    b = 0

    for i in range(len(PMmsg)):
        if diff_PMIID[i] == 0:
            while len(Process) <= a:
                Process.append([])  # Expand the list with a new sublist
            Process[a].append(PMmsg[i])
            b += 1
        else:
            a += 1
            b = 0

    # Convert the list of lists into a numpy array, padding shorter sublists with None
    max_length = max(len(sublist) for sublist in Process)
    Process = np.array([sublist + [None]*(max_length - len(sublist)) for sublist in Process])

    # Transpose the array so it matches the expected structure
    Process = Process.T
    szP = Process.shape

    # Grouping columns with unique values in the first row
    first_row = [x for x in Process[0, :] if x is not None]

    # Find unique values in the first row
    unique_values = np.unique(first_row)

    grouped_columns = []
    for value in unique_values:
        column_indices = np.where(Process[0, :] == value)[0]
        grouped_columns.append(Process[:, column_indices])

    colors = plt.get_cmap('tab10', len(unique_values))

    # Event processing section
    Events = np.unique(PMmsg)
    EventSamples = []
    a = 1
    b = 0
    c = 0
    diff_PMmsg = np.diff(PMmsg, prepend=0)

    for i in range(len(PMmsg)):
        if diff_PMmsg[i] != 0:
            EventSamples.append([PMID[i], PMmsg[i], a])
            b += 1
            a = 1
        else:
            a += 1

    EventSamples = np.array(EventSamples)

    # Define the number of nodes based on the number of events
    num_nodes = len(Events)

    # Initialize edges array
    edges = []

    for i in range(1, len(EventSamples)):
        edges.append([EventSamples[i-1, 1]+1, EventSamples[i, 1]+1])

    edges = np.array(edges)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    print(counts)
    
    # Create the graph object
    G = nx.DiGraph()
    for i, (edge, count) in enumerate(zip(edges, counts)):
        print(count)
        G.add_edge(edge[0], edge[1], weight=count)

    # Set linewidths and colors based on the difference between consecutive nodes
    linewidths = []
    arrow_sizes = []
    edge_colors = []
    ColorCodes = []

    for i in range(len(edges)):
        if edges[i, 1] - edges[i, 0] == 1 or ((edges[i, 1] - edges[i, 0]) == min(Events)-max(Events)):
            linewidths.append(0.5)
            edge_colors.append([0, 0, 1])  # Dark blue color
            arrow_sizes.append(10)
            ColorCodes.append(1)
        else:
            linewidths.append(0.5)
            arrow_sizes.append(7)
            edge_colors.append([1, 0.5, 1])  # Light blue color
            ColorCodes.append(2) 

    ColorCodes_array = np.array(ColorCodes)
    matrix = np.column_stack((counts, ColorCodes_array))
    print(matrix)

    # Generate a stable, skeleton-like layout
    pos = nx.spectral_layout(G)

    # Plot the graph with curved edges
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='r', node_size=100)

    # Draw the curved edges with arrows using the custom function
    draw_curved_edges(G, pos, ax, edge_labels=nx.get_edge_attributes(G, 'weight'), 
                      edge_colors=ColorCodes, linewidths=linewidths, arrow_sizes=arrow_sizes)

    # Adjust node labels
    num_nodes = max(max(edges[:, 0]), max(edges[:, 1]))
    labels = {i: str(i-1) for i in range(1, num_nodes + 1)}
    nx.draw_networkx_labels(G, pos, labels, font_color='w', font_size=6)

    plt.title('Skeleton Graph - Curvy S-Shape with Custom Linewidths and Colors')
    plt.grid(True)

    # Save the plot as a file
    plot_filename = "skeleton_graph.png"
    fig.savefig(plot_filename)
    #plt.close(fig)

    SkippedNodeAnomally = np.hstack((edges, counts.reshape(-1, 1)))

    max_first_column = np.max(SkippedNodeAnomally[:, 0])
    max_counts = np.max(counts)


    new_column = np.where(
        SkippedNodeAnomally[:, 1] - SkippedNodeAnomally[:, 0] < 0, 
        SkippedNodeAnomally[:, 1] + max_first_column - SkippedNodeAnomally[:, 0], 
        SkippedNodeAnomally[:, 1] - SkippedNodeAnomally[:, 0]
    )

    SkippedNodeAnomally = np.hstack((SkippedNodeAnomally[:, :2], new_column.reshape(-1, 1), SkippedNodeAnomally[:, 2].reshape(-1, 1)))

    new_matrix = SkippedNodeAnomally[SkippedNodeAnomally[:, 2] != 1]


    adjusted_second_column = np.where(
        new_matrix[:, 1] - new_matrix[:, 0] < 0, 
        new_matrix[:, 1] + max_first_column, 
        new_matrix[:, 1]
    )

    new_matrix[:, 1] = adjusted_second_column
    new_list = [np.arange(row[0] + 1, row[1]) for row in new_matrix]
    new_array = np.array(new_list, dtype=object)

    fourth_column = new_matrix[:, 3]
    result_array = np.array(list(zip(new_list, fourth_column)), dtype=object)

    final_list = []

    for row in result_array:
        first_column_values = row[0]
        second_column_value = row[1]
        
        for value in first_column_values:
            final_list.append([value, second_column_value])

    FinalMatrix = np.array(final_list)


    unique_values, indices = np.unique(FinalMatrix[:, 0], return_index=True)
    summed_values = {}

    for value in unique_values:
        summed_values[value] = FinalMatrix[FinalMatrix[:, 0] == value, 1].astype(int).sum()

    FinalMatrix = np.array([[key, summed_values[key]] for key in summed_values])

    third_column = FinalMatrix[:, 1].astype(float) * (1 / max_counts)

    FinalMatrix = np.hstack((FinalMatrix, third_column.reshape(-1, 1)))


    # Sample mean and anomaly detection
    EventSamples2 = []
    SampleMean = []

    for event in Events:
        Vec = []
        for a in range(len(EventSamples)):
            if EventSamples[a, 1] == event:
                Vec.append(EventSamples[a, 2])
        
        if Vec:  # Check if Vec is not empty
            EventSamples2.append(Vec)
            SampleMean.append([event, np.mean(Vec)])

    # Convert EventSamples2 to a 2D numpy array, where each row corresponds to an event's samples
    EventSamples2 = np.array([np.array(event_samples) for event_samples in EventSamples2], dtype=object)

    # Calculate the standard deviation for each event's samples
    Samplestd = np.array([np.std(event_samples) for event_samples in EventSamples2])

    # Z-score calculation
    zScoreData = []
    SampleAnomally = []
    EventMean = []

    AnomallyLimit = 2

    for i in range(len(EventSamples)):
        step = int(EventSamples[i, 1])
        mean_value = SampleMean[step][1]  # Adjusted for list indexing
        zScore = (EventSamples[i, 2] - mean_value) / Samplestd[step]
        zScoreData.append(zScore)
        EventMean.append(mean_value)

        if zScore > AnomallyLimit:
            SampleAnomally.append(1)
        else:
            SampleAnomally.append(0)

    EventSamples = np.column_stack([EventSamples, EventMean, zScoreData, SampleAnomally])
    AnomallyEvents = EventSamples[EventSamples[:, -1] != 0]

    for i in tree.get_children():
        tree.delete(i)

    # Populate the Treeview with AnomallyEvents
    for row in AnomallyEvents:
        tree.insert("", "end", values=tuple(row))

    # Create the PDF document with landscape orientation
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputfile = "ProcessMiningReport"
    pdf_filename = f"{outputfile}_{current_datetime}.pdf"
    
    doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))
    elements = []

    # Adjust the image size to fit within the available frame size
    # Calculate the image size to be within the available frame dimensions
    max_width, max_height = 1636, 456  # Set maximum width and height
    img_width, img_height = Image(plot_filename)._restrictSize(max_width, max_height)

    # Add the plot image to the PDF with adjusted size
    elements.append(Image(plot_filename, width=img_width, height=img_height))

    # Convert AnomallyEvents to a list of lists
    table_data = [["PMID", "PMmsg", "Duration", "Mean", "Z-Score", "Anomaly"]]
    for row in AnomallyEvents:
        table_data.append([str(x) for x in row])

    # Create a table for AnomallyEvents
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, rl_colors.black),
    ]))

    elements.append(table)

    for i in treeFM.get_children():
        treeFM.delete(i)

    # Populate the Treeview with AnomallyEvents
    for row in FinalMatrix:
        treeFM.insert("", "end", values=tuple(row))
    
    FinalMatrixTable_data = [["Event", "Count", "Probability"]] 
    for row in FinalMatrix:
        FinalMatrixTable_data.append([str(x) for x in row])

    # Create a table for FinalMatrix
    FinalMatrixTable = Table(FinalMatrixTable_data)
    FinalMatrixTable.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, rl_colors.black),
    ]))

    elements.append(FinalMatrixTable)

    # Build the PDF
    doc.build(elements)

    messagebox.showinfo("PDF Created", f"PDF file '{pdf_filename}' created successfully.")
# Create the main window
root = tk.Tk()
root.title("Excel File Selector")

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.grid(row=0, column=0, padx=5, pady=5, sticky="w")

# Create an upload button and place it in the button frame
upload_button = tk.Button(button_frame, text="Upload Excel File", command=upload_file)
upload_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

# Create a button to select the latest file from the hardcoded folder and place it in the button frame
latest_file_button = tk.Button(button_frame, text="Select Latest Excel File from Folder", command=select_latest_file)
latest_file_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# Create a label to display the status of the file selection
label = tk.Label(root, text="No file selected", pady=5, padx=5, anchor="w")
label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# Create a Listbox to display the headers
headers_listbox = tk.Listbox(root, selectmode=tk.SINGLE, height=10, width=50)
headers_listbox.grid(row=2, column=0, padx=5, pady=5, sticky="w")

# Create a frame for the PMmsg and PMID assignment buttons
assign_frame = tk.Frame(root)
assign_frame.grid(row=3, column=0, padx=5, pady=5, sticky="w")

# Create buttons to assign the selected header to PMmsg and PMID
assign_pmmsg_button = tk.Button(assign_frame, text="Assign to PMmsg", command=assign_pmmsg)
assign_pmmsg_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

assign_pmid_button = tk.Button(assign_frame, text="Assign to PMID", command=assign_pmid)
assign_pmid_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")

# Labels to display the current PMmsg and PMID assignments
pmmsg_label = tk.Label(root, text="PMmsg: Not Assigned", pady=5, padx=5, anchor="w")
pmmsg_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

pmid_label = tk.Label(root, text="PMID: Not Assigned", pady=5, padx=5, anchor="w")
pmid_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

# Create a button to run the analysis after selections
run_analysis_button = tk.Button(root, text="Run Analysis", command=run_analysis)
run_analysis_button.grid(row=6, column=0, padx=5, pady=5, sticky="w")

# Create a frame for the AnomallyEvents Treeview
tree_frame = tk.Frame(root)
tree_frame.grid(row=0, column=1, rowspan=7, padx=5, pady=5, sticky="nsew")

# Add a vertical scrollbar for the AnomallyEvents Treeview
tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
tree_scrollbar.pack(side="right", fill="y")

# Create the AnomallyEvents Treeview
tree = ttk.Treeview(tree_frame, columns=("PMID", "PMmsg", "Duration", "Mean", "Z-Score", "Anomaly"), show='headings', yscrollcommand=tree_scrollbar.set)
tree.heading("PMID", text="PMID")
tree.heading("PMmsg", text="PMmsg")
tree.heading("Duration", text="Duration")
tree.heading("Mean", text="Mean")
tree.heading("Z-Score", text="Z-Score")
tree.heading("Anomaly", text="Anomaly")
tree.pack(fill=tk.BOTH, expand=True)

# Attach the scrollbar to the Treeview
tree_scrollbar.config(command=tree.yview)

# Create a frame for the Final Matrix Treeview
tree_frameFM = tk.Frame(root)
tree_frameFM.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

# Add a vertical scrollbar for the Final Matrix Treeview
tree_scrollbarFM = ttk.Scrollbar(tree_frameFM, orient="vertical")
tree_scrollbarFM.pack(side="right", fill="y")

# Create the Final Matrix Treeview
treeFM = ttk.Treeview(tree_frameFM, columns=("Event", "Count", "Probability"), show='headings', yscrollcommand=tree_scrollbarFM.set)
treeFM.heading("Event", text="Event")
treeFM.heading("Count", text="Count")
treeFM.heading("Probability", text="Probability")
treeFM.pack(fill=tk.BOTH, expand=True)

# Attach the scrollbar to the Treeview
tree_scrollbarFM.config(command=treeFM.yview)

# Ensure proper resizing behavior
root.grid_rowconfigure(7, weight=1)
root.grid_columnconfigure(1, weight=1)
tree_frame.grid_rowconfigure(0, weight=1)
tree_frame.grid_columnconfigure(0, weight=1)
tree_frameFM.grid_rowconfigure(0, weight=1)
tree_frameFM.grid_columnconfigure(0, weight=1)

root.mainloop()
