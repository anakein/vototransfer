import plotly.graph_objects as go
import pandas as pd
import os

def generate_sankey(transition_matrix_df, title, output_path, start_label="Start", end_label="End"):
    """
    Generates a Sankey diagram from a transition matrix and saves it to HTML.
    """
    print(f"Generating Sankey for {title}...")
    
    sources = transition_matrix_df.index.tolist()
    targets = transition_matrix_df.columns.tolist()
    
    # Create node list (Sources first, then Targets)
    all_nodes = [f"{s} ({start_label})" for s in sources] + [f"{t} ({end_label})" for t in targets]
    node_indices = {name: i for i, name in enumerate(all_nodes)}
    
    # Create links
    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []
    
    # Define colors for parties
    party_colors = {
        'PSOE': 'red', 'PP': 'blue', 'Cs': 'orange', 'VOX': 'green', 
        'IU': 'purple', 'Podemos': 'purple', 
        'Adelante Andalucía': 'mediumpurple', 'AA': 'mediumpurple',
        'AxSi': 'darkturquoise',
        'PA': 'teal', 'CA': 'teal', # CA is Coalición Andalucista
        'NA': 'darkgreen',
        'PCPA': 'darkred', 'IZAR': 'firebrick',
        'RISA': 'gold', 'PRAO': 'lightseagreen', 'PNdeA': 'olive',
        'Abstencion': 'grey', 'Otros': 'lightgrey'
    }
    
    # Helper to clean party name from label "PSOE (start)" -> "PSOE"
    def get_color(label):
        clean_name = label.split(' (')[0]
        return party_colors.get(clean_name, 'silver')

    node_colors = [get_color(name) for name in all_nodes]
    
    # Create links
    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []
    
    for src in sources:
        for tgt in targets:
            val = transition_matrix_df.loc[src, tgt]
            if val > 0.005: # Filter tiny links < 0.5%
                link_sources.append(node_indices[f"{src} ({start_label})"])
                link_targets.append(node_indices[f"{tgt} ({end_label})"])
                link_values.append(val)
                
                # Link color: Source color with opacity
                src_color = party_colors.get(src, 'silver')
                # Simple heuristic for rgba if named color
                # For simplicity, let's rely on plotly handling named colors with opacity if we use rgba.
                # But mapping 'red' to 'rgba(255,0,0,0.5)' is tedious.
                # Let's use a mapping for common ones or let plotly handle it by just assigning the color string.
                # To make it readable, usually simple 'red' is too strong for links.
                # We will define a specific opacity map or just string manipulation if possible.
                # Actually, Plotly suggests 'rgba(...)'. 
                
                # Let's treat it simple: use the source color. 
                # If the user wants readability, distinct link colors help.
                # If we leave it solid it mimics the node.
                # Let's try to pass the node color directly.
                link_colors.append(src_color)
                
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = all_nodes,
          color = node_colors,
          hovertemplate = '%{label}<br>Total: %{value:.1%}<extra></extra>' # Custom node hover
        ),
        link = dict(
          source = link_sources, 
          target = link_targets,
          value = link_values,
          color = link_colors,
          hovertemplate = 'Origen: %{source.label}<br>Destino: %{target.label}<br>Transferencia: %{value:.1%}<extra></extra>' # Custom link hover
      ))])
    
    # Update layout to support percentages in values if possible (Sankey values are absolute usually, but here we pass 0-1)
    # We passed 0-1 floats. Plotly handles them. 
    # To display as % in hover, we added the format in hovertemplate.
    
    fig.update_layout(title_text=title, font_size=12)
    fig.write_html(output_path)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    # Test stub
    pass
