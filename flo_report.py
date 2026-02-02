import dotenv
import pendulum
from sqlalchemy import asc, cast
from sqlalchemy import create_engine, select, BigInteger, MetaData, Table
from sqlalchemy.orm import sessionmaker
from config import Settings
from models import MessageSql
from gridflo.asl.types import FloParamsHouse0
from gridflo.dijkstra_types import DNode, DEdge
from gridflo import Flo, DGraphVisualizer, DNodeVisualizer
from gridflo.asl.types import WinterOakSupergraphParams
from gridflo.supergraph_generator import SupergraphGenerator, RuleBasedStorageModel
from gridflo.supergraph_generator import WinterOakSupergraphParams
from gridflo.dijkstra_types import DNode
from gridflo.dgraph_visualizer import DNodeComparator
import gc
import os
import shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image

print("\nWelcome to the FLO report generator!\n")
house_alias = input("Enter house alias: ")

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

# TEMPORARY
# house_alias = "oak"
# start_time = pendulum.datetime(2026, 1, 29, 20, tz='America/New_York')
# end_time = pendulum.datetime(2026, 1, 29, 22, tz='America/New_York')

start_ms = start_time.timestamp()*1000
end_ms = end_time.timestamp()*1000
print(f"Generating report for {house_alias} from {start_time} to {end_time}\n")

# ---------------------------------------------------
# Part 1: Find FLO params messages
# ---------------------------------------------------

stmt = select(MessageSql).filter(
    MessageSql.message_type_name == "flo.params.house0",
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
    MessageSql.message_persisted_ms <= cast(int(end_ms+10*60*1000), BigInteger),
    MessageSql.message_persisted_ms >= cast(int(start_ms-10*60*1000), BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))

settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
result = session.execute(stmt)
messages = result.scalars().all()

flo_params_messages: list[MessageSql] = []
for m in messages:
    if pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York').minute == 57:
        print(f"Adding message from {m.from_alias} at {pendulum.from_timestamp(m.message_persisted_ms/1000, tz='America/New_York')}")
        flo_params_messages.append(m)

print(f"Found {len(messages)} messages and {len(flo_params_messages)} of them at minute 57\n")

session.close()
engine.dispose()

# ---------------------------------------------------
# Part 2: Find hourly data: hp_elec_in, hp_heat_out
# ---------------------------------------------------

gbo_engine = create_engine(settings.gbo_db_url_no_async.get_secret_value())
hourly_electricity = Table('hourly_electricity', MetaData(), autoload_with=gbo_engine)
GboSession = sessionmaker(bind=gbo_engine)
gbo_session = GboSession()
stmt_hourly = select(hourly_electricity).where(
    hourly_electricity.c.short_alias == house_alias,
    hourly_electricity.c.hour_start_s >= int(start_ms // 1000),
    hourly_electricity.c.hour_start_s <= int(end_ms // 1000),
).order_by(asc(hourly_electricity.c.hour_start_s))
hourly_records = gbo_session.execute(stmt_hourly).all()

hourly_hour_start_s = []
hourly_hp_elec_in = []
hourly_hp_heat_out = []
for rec in hourly_records:
    hourly_hour_start_s.append(rec.hour_start_s)
    hourly_hp_elec_in.append(getattr(rec, 'hp_elec_in', getattr(rec, 'hp_kwh_el', 0)))
    hourly_hp_heat_out.append(getattr(rec, 'hp_kwh_th', getattr(rec, 'hp_heat_out', 0)))
print(f"Found {len(hourly_records)} hourly records for {house_alias} (hp_elec_in, hp_heat_out)")
gbo_session.close()
gbo_engine.dispose()

# ---------------------------------------------------
# Part 3: Generate plots in plots/ directory
# ---------------------------------------------------

if os.path.exists('plots'):
    shutil.rmtree('plots')
os.makedirs('plots', exist_ok=True)

true_init_energy, true_final_energy = [0]*len(flo_params_messages), [0]*len(flo_params_messages)
heat_to_store_expected = []
heat_from_hp_expected = []
true_initial_states, true_final_states = [None]*len(flo_params_messages), [None]*len(flo_params_messages)

for i, flo_params_msg in enumerate(flo_params_messages):
    flo_params = FloParamsHouse0(**flo_params_msg.payload)
    g = Flo(flo_params.to_bytes())
    g.solve_dijkstra()
    g.generate_recommendation(flo_params.to_bytes())
    v = DGraphVisualizer(g)
    v.plot(show=False,save_as=f'plots/flo{i+1}_graph.png')
    v.plot_pq_pairs(save_as=f'plots/flo{i+1}_pq_pairs.png')
    initial_node_edge: DEdge = [e for e in g.bid_edges[g.initial_node] if e.head == g.initial_node.next_node][0]
    hp_heat_out_expected = initial_node_edge.hp_heat_out
    heat_to_store_expected.append(hp_heat_out_expected - initial_node_edge.load_and_losses)
    heat_from_hp_expected.append(hp_heat_out_expected)

    winter_oak_supergraph_params = WinterOakSupergraphParams(
        num_layers=flo_params.num_layers,
        storage_volume_gallons=flo_params.storage_volume_gallons,
        hp_max_elec_kw=flo_params.hp_max_elec_kw,
        cop_intercept=flo_params.cop_intercept,
        cop_oat_coeff=flo_params.cop_oat_coeff,
        cop_min=flo_params.cop_min,
        cop_min_oat_f=flo_params.cop_min_oat_f,
        constant_delta_t=flo_params.constant_delta_t,
    )
    
    true_initial_node = DNode(
        top_temp=flo_params.initial_top_temp_f,
        middle_temp=flo_params.initial_middle_temp_f,
        bottom_temp=flo_params.initial_bottom_temp_f,
        thermocline1=flo_params.initial_thermocline_1,
        thermocline2=flo_params.initial_thermocline_2,
        parameters=winter_oak_supergraph_params,
    )
    true_initial_states[i] = true_initial_node

    true_init_energy[i] = true_initial_node.energy
    true_init_node = DNodeVisualizer(true_initial_node, 'true_initial')
    init_node = DNodeVisualizer(g.initial_node, 'initial')
    expected_node = DNodeVisualizer(g.initial_node.next_node, 'expected')
    true_init_node.plot(save_as=f'plots/flo{i+1}_true_initial.png')
    init_node.plot(save_as=f'plots/flo{i+1}_initial.png')
    expected_node.plot(save_as=f'plots/flo{i+1}_expected.png')
    if i!=0:
        true_final_node = DNode(
            top_temp=flo_params.initial_top_temp_f,
            middle_temp=flo_params.initial_middle_temp_f,
            bottom_temp=flo_params.initial_bottom_temp_f,
            thermocline1=flo_params.initial_thermocline_1,
            thermocline2=flo_params.initial_thermocline_2,
            parameters=winter_oak_supergraph_params,
        )
        true_final_states[i-1] = true_final_node
        true_final_energy[i-1] = true_final_node.energy
        true_final_node = DNodeVisualizer(true_final_node, 'true_final')
        true_final_node.plot(save_as=f'plots/flo{i}_true_final.png')
        final_node = DNodeVisualizer(g.initial_node, 'final')
        final_node.plot(save_as=f'plots/flo{i}_final.png')

    del g, v, init_node, expected_node
    gc.collect()

heat_to_store_true = [final-init for final, init in zip(true_final_energy, true_init_energy)]
heat_from_hp_true = []
for i in range(len(flo_params_messages)):
    start_s = flo_params_messages[i].message_persisted_ms / 1000
    end_s = flo_params_messages[i + 1].message_persisted_ms / 1000 if i + 1 < len(flo_params_messages) else end_ms / 1000
    total = sum(hourly_hp_heat_out[j] for j in range(len(hourly_hour_start_s)) if start_s <= hourly_hour_start_s[j] < end_s)
    heat_from_hp_true.append(total)

for i in range(len(flo_params_messages)):
    print(f"Heat to store true: {heat_to_store_true[i]}, Heat to store expected: {heat_to_store_expected[i]}")

for j in range(len(hourly_hour_start_s)):
    hour_dt = pendulum.from_timestamp(hourly_hour_start_s[j], tz='America/New_York')
    print(f"Hourly {hour_dt}: hp_elec_in={hourly_hp_elec_in[j]}, hp_heat_out={hourly_hp_heat_out[j]}")

# ---------------------------------------------------
# Comparison plot
# ---------------------------------------------------

params = WinterOakSupergraphParams(
    num_layers=27,
    storage_volume_gallons=360,
    hp_max_elec_kw=11.0,
    cop_intercept=1.02,
    cop_oat_coeff=0.0257,
    cop_min=1.4,
    cop_min_oat_f=15.0,
    constant_delta_t=20
)

generator = SupergraphGenerator(params)
generator.create_nodes()

model = RuleBasedStorageModel(
    params,
    generator.nodes,
    generator.nodes_by,
    generator.logger
)

for i in range(len(flo_params_messages)-1):
    start_node = generator.find_closest_node(true_initial_states[i])
    predicted_end_state_from_true = model.next_node(true_initial_states[i], heat_to_store_true[i])
    predicted_end_state_from_node = model.next_node(start_node, heat_to_store_true[i])
    end_node = generator.find_closest_node(predicted_end_state_from_node)
    DNodeComparator(
        nodes = [
            true_initial_states[i], 
            start_node, 
            predicted_end_state_from_true,
            predicted_end_state_from_node, 
            end_node, 
            true_final_states[i], 
        ],
        titles = [
            'True start',
            'Node start',
            'Predicted final\nfrom true start',
            'Predicted final\nfrom node start',
            'Node final',
            'True final',
        ],
        rows = [
            1, 1, 
            2, 2, 2, 2
        ],
        cols = [
            2, 3,
            2, 3, 4, 1
        ],
        heat_to_store = round(heat_to_store_expected[i], 1)
    ).plot(save_as=f'plots/flo{i}_node_comparison.png')

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

# Right side: node comparison plot
right_width = 1.8 * inch  # Fixed width for right side comparison plot
right_x = left_margin + left_width + left_gap

# Calculate height per graph (to fit 4 per page)
usable_height = page_height - 2 * margin
graph_height = (usable_height - 2 * (graphs_per_page - 1) * 0.2 * inch) / graphs_per_page

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
        
        # Draw node comparison plot on right side (where node plots were)
        comparison_path = f'plots/flo{graph_num - 1}_node_comparison.png'
        comp_y = None
        comp_display_height = None
        if graph_num < num_graphs and os.path.exists(comparison_path):
            comp_img = Image.open(comparison_path)
            comp_width, comp_height = comp_img.size
            comp_aspect_ratio = comp_height / comp_width
            if graph_y is not None and graph_display_height is not None:
                comp_display_height = min(graph_display_height, right_width * comp_aspect_ratio)
            else:
                comp_display_height = min(graph_height, right_width * comp_aspect_ratio)
            comp_display_width = min(right_width, comp_display_height / comp_aspect_ratio)
            # Position comparison higher (move up by ~0.8 inch)
            comp_y = ((graph_y - comp_display_height) if (graph_y is not None) else (current_y - comp_display_height)) + 0.8 * inch
            comp_x = right_x + (right_width - comp_display_width) / 2

            # Draw heat text above the comparison plot first
            if graph_num <= len(heat_to_store_true) and graph_num > 0:
                heat_text = f"Heat to store - True: {heat_to_store_true[graph_num-1]:.1f} kWh, Expected: {heat_to_store_expected[graph_num-1]:.1f} kWh"
                c.setFont("Helvetica", 4)
                text_y = comp_y + comp_display_height + 0.08 * inch
                text_x = right_x + 0.1 * inch
                c.drawString(text_x, text_y, heat_text)
                if graph_num <= len(heat_from_hp_true) and graph_num <= len(heat_from_hp_expected):
                    heat_from_hp_text = f"Heat from HP - True: {heat_from_hp_true[graph_num-1]:.1f} kWh, Expected: {heat_from_hp_expected[graph_num-1]:.1f} kWh"
                    c.drawString(text_x, text_y - 0.12 * inch, heat_from_hp_text)

            c.drawImage(comparison_path, comp_x, comp_y,
                        width=comp_display_width, height=comp_display_height)

c.save()
print(f"PDF report saved as {pdf_path}")
if os.path.exists('plots'):
    shutil.rmtree('plots')