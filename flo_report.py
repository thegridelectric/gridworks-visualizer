import dotenv
import pendulum
from sqlalchemy import asc, cast
from sqlalchemy import create_engine, select, BigInteger
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select, BigInteger
from config import Settings
from models import MessageSql
from gridflo.asl.types import FloParamsHouse0
from gridflo import Flo, DGraphVisualizer, DNodeVisualizer
import gc
import os
import shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image

print("\nWelcome to the FLO report generator!\n")
house_alias = input("Enter house alias: ")
if not house_alias:
    print("House alias is required")
    exit()

message_type = "flo.params.house0"
now = pendulum.now(tz='America/New_York')
yesterday_8pm = now.subtract(days=1).set(hour=20, minute=0, second=0, microsecond=0)

start_input = input("Enter start year, month, day, hour (default: yesterday 8pm): ")
if start_input and len(start_input.split(',')) == 4:
    START_YEAR, START_MONTH, START_DAY, START_HOUR = start_input.split(',')
    START_YEAR, START_MONTH, START_DAY, START_HOUR = int(START_YEAR), int(START_MONTH), int(START_DAY), int(START_HOUR)
else:
    START_YEAR, START_MONTH, START_DAY, START_HOUR = yesterday_8pm.year, yesterday_8pm.month, yesterday_8pm.day, yesterday_8pm.hour

end_input = input("Enter end year, month, day, hour (default: now): ")
if end_input and len(end_input.split(',')) == 4:
    END_YEAR, END_MONTH, END_DAY, END_HOUR = end_input.split(',')
    END_YEAR, END_MONTH, END_DAY, END_HOUR = int(END_YEAR), int(END_MONTH), int(END_DAY), int(END_HOUR)
else:
    END_YEAR, END_MONTH, END_DAY, END_HOUR = now.year, now.month, now.day, now.hour

start_time = pendulum.datetime(START_YEAR, START_MONTH, START_DAY, START_HOUR, tz='America/New_York')
end_time = pendulum.datetime(END_YEAR, END_MONTH, END_DAY, END_HOUR, tz='America/New_York')
start_ms = start_time.timestamp()*1000
end_ms = end_time.timestamp()*1000
print(f"Generating report for {house_alias} from {start_time} to {end_time}\n")

# ---------------------------------------------------
# Part 1: Find FLO params messages
# ---------------------------------------------------

stmt = select(MessageSql).filter(
    MessageSql.message_type_name == message_type,
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
    MessageSql.message_persisted_ms <= cast(int(end_ms-10*60*1000), BigInteger),
    MessageSql.message_persisted_ms >= cast(int(start_ms-10*60*1000), BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
result = session.execute(stmt)
messages = result.scalars().all()

flo_params_messages = []
for m in messages:
    if pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York').minute == 57:
        print(f"Adding message from {m.from_alias} at {pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York')}")
        flo_params_messages.append(m)

print(f"Found {len(messages)} messages and {len(flo_params_messages)} of them at minute 57")

session.close()
engine.dispose()

# ---------------------------------------------------
# Part 2: Generate plots in plots/ directory
# ---------------------------------------------------

if os.path.exists('plots'):
    shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

for i, flo_params_msg in enumerate(flo_params_messages):
    flo_params = FloParamsHouse0(**flo_params_msg.payload)
    g = Flo(flo_params.to_bytes())
    g.solve_dijkstra()
    g.generate_recommendation(flo_params.to_bytes())
    v = DGraphVisualizer(g)
    if i!=0:
        final_node = DNodeVisualizer(g.initial_node, 'final')
        final_node.plot(save_as=f'plots/flo{i}_final.png')
    v.plot(show=False,save_as=f'plots/flo{i+1}_graph.png')
    v.plot_pq_pairs(save_as=f'plots/flo{i+1}_pq_pairs.png')
    
    init_node = DNodeVisualizer(g.initial_node, 'initial')
    expected_node = DNodeVisualizer(g.initial_node.next_node, 'expected')
    init_node.plot(save_as=f'plots/flo{i+1}_initial.png')
    expected_node.plot(save_as=f'plots/flo{i+1}_expected.png')

    del g, v, init_node, expected_node
    gc.collect()

# ---------------------------------------------------
# Part 3: Generate PDF report
# ---------------------------------------------------

pdf_path = os.path.expanduser(f'~/Desktop/flo_report_{house_alias}.pdf')
if os.path.exists(pdf_path):
    os.remove(pdf_path)

print("Generating PDF report...")
c = canvas.Canvas(pdf_path, pagesize=A4)
page_width, page_height = A4

# Calculate dimensions
margin = 0.5 * inch
graphs_per_page = 4

# Left side: graphs
left_margin = margin
left_width = 5.5 * inch  # Fixed width for left side graphs (increased)
left_gap = 0.2 * inch  # Gap between left and right sides (reduced)

# Right side: node plots (initial, expected, final)
right_width = 1.8 * inch  # Fixed width for right side plots (reduced)
right_x = left_margin + left_width + left_gap

# Calculate height per graph (to fit 4 per page)
usable_height = page_height - 2 * margin
graph_height = (usable_height - 2 * (graphs_per_page - 1) * 0.2 * inch) / graphs_per_page

# Fixed dimensions for node plots (arranged horizontally)
node_plot_height = 1.2 * inch  # Fixed height for each node plot (reduced)
node_plot_gap = 0.05 * inch  # Gap between node plots horizontally (reduced)
node_plot_width = (right_width - 2 * node_plot_gap) / 3  # Width for each of the 3 plots

num_graphs = len(flo_params_messages)

for page_start in range(0, num_graphs, graphs_per_page):
    if page_start > 0:
        c.showPage()
    
    y_position = page_height - margin
    
    # Process up to 4 graphs on this page
    for graph_idx in range(graphs_per_page):
        graph_num = page_start + graph_idx + 1
        
        if graph_num > num_graphs:
            break
        
        # Calculate y position for this graph
        current_y = y_position - graph_idx * (graph_height + 0.2 * inch)
        
        # Draw graph on left side
        graph_path = f'plots/flo{graph_num}_graph.png'
        graph_y = None
        graph_display_height = None
        if os.path.exists(graph_path):
            img = Image.open(graph_path)
            img_width, img_height = img.size
            aspect_ratio = img_height / img_width
            
            # Scale graph to fit the left width while maintaining aspect ratio
            graph_display_height = min(graph_height, left_width * aspect_ratio)
            graph_display_width = min(left_width, graph_display_height / aspect_ratio)
            
            # Center vertically in the allocated space
            graph_y = current_y - graph_display_height
            
            c.drawImage(graph_path, left_margin, graph_y,
                       width=graph_display_width, height=graph_display_height)
        
        # Draw node plots on right side (initial, expected, final side by side)
        node_plots = []
        
        init_path = f'plots/flo{graph_num}_initial.png'
        if os.path.exists(init_path):
            node_plots.append(init_path)
        
        expected_path = f'plots/flo{graph_num}_expected.png'
        if os.path.exists(expected_path):
            node_plots.append(expected_path)
        
        final_path = f'plots/flo{graph_num}_final.png'
        if os.path.exists(final_path):
            node_plots.append(final_path)
        
        # Draw node plots side by side horizontally, aligned with middle/bottom of graph
        node_x = right_x
        # Position at middle-bottom of graph (bottom of graph + some offset toward middle)
        if graph_y is not None and graph_display_height is not None:
            # Align bottom of node plots closer to bottom of graph
            node_y = graph_y + graph_display_height * 0.15 - node_plot_height
        else:
            # Fallback if graph doesn't exist
            node_y = current_y - node_plot_height
        
        for img_path in node_plots:
            img = Image.open(img_path)
            img_width, img_height = img.size
            aspect_ratio = img_height / img_width
            
            # Scale to fit fixed dimensions while maintaining aspect ratio
            if node_plot_width * aspect_ratio <= node_plot_height:
                # Width is limiting factor
                plot_display_width = node_plot_width
                plot_display_height = node_plot_width * aspect_ratio
            else:
                # Height is limiting factor
                plot_display_height = node_plot_height
                plot_display_width = node_plot_height / aspect_ratio
            
            # Center vertically in the allocated space
            plot_y = node_y + (node_plot_height - plot_display_height) / 2
            
            c.drawImage(img_path, node_x, plot_y,
                       width=plot_display_width, height=plot_display_height)
            
            node_x += node_plot_width + node_plot_gap
        
        # Draw pq_pairs plot on the right, positioned above the node plots
        pq_pairs_path = f'plots/flo{graph_num}_pq_pairs.png'
        if os.path.exists(pq_pairs_path):
            pq_img = Image.open(pq_pairs_path)
            pq_img_width, pq_img_height = pq_img.size
            pq_aspect_ratio = pq_img_height / pq_img_width
            
            # Use full right width for the plot
            pq_plot_width = right_width
            pq_plot_height = pq_plot_width * pq_aspect_ratio
            
            # Cap the height to avoid overlapping with row above
            max_pq_height = 1.4 * inch
            if pq_plot_height > max_pq_height:
                pq_plot_height = max_pq_height
                pq_plot_width = pq_plot_height / pq_aspect_ratio
            
            # Position above the node plots row, centered horizontally in right area
            pq_x = right_x + (right_width - pq_plot_width) / 2
            pq_y = node_y + node_plot_height + 0.05 * inch
            
            c.drawImage(pq_pairs_path, pq_x, pq_y,
                       width=pq_plot_width, height=pq_plot_height)

c.save()
print(f"PDF report saved as {pdf_path}")
if os.path.exists('plots'):
    shutil.rmtree('plots')