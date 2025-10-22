from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
import io

app = Flask(__name__)

# Energy ranges for analysis
ENERGY_RANGES = {
    'R1': (2, 4),
    'R2': (7, 13),
    'R3': (80, 100),
    'R4': (900, 1100),
    'R5': (1400, 1600)
}

def parse_csv(file_content):
    """Parse the CSV content and return a DataFrame"""
    # Try comma separator first, then tab
    try:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), 
                         sep=',', 
                         header=None,
                         names=['invariant_mass', 'particle_type', 'combination'])
    except:
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), 
                         sep='\t', 
                         header=None,
                         names=['invariant_mass', 'particle_type', 'combination'])
    
    # Remove rows with empty values in first two columns
    df = df.dropna(subset=['invariant_mass', 'particle_type'])
    
    # Convert invariant_mass to float
    df['invariant_mass'] = pd.to_numeric(df['invariant_mass'], errors='coerce')
    
    # Handle particle_type - convert to int, keep combination as string
    df['particle_type'] = pd.to_numeric(df['particle_type'], errors='coerce')
    
    # Drop rows where conversion failed
    df = df.dropna(subset=['invariant_mass', 'particle_type'])
    
    # Convert particle_type to int
    df['particle_type'] = df['particle_type'].astype(int)
    
    # Fill NaN in combination column with empty string
    df['combination'] = df['combination'].fillna('').astype(str).str.strip()
    
    return df

def calculate_statistics(df, particle_types):
    """Calculate statistics for energy ranges"""
    # Filter by particle_type
    filtered_df = df[df['particle_type'].isin(particle_types)]
    
    stats = []
    total_events = len(filtered_df)
    
    for range_name, (min_e, max_e) in ENERGY_RANGES.items():
        range_df = filtered_df[(filtered_df['invariant_mass'] >= min_e) & 
                               (filtered_df['invariant_mass'] <= max_e)]
        events = len(range_df)
        mean = range_df['invariant_mass'].mean() if events > 0 else 0
        stats.append({
            'range': f"{range_name} ({min_e}-{max_e} GeV)",
            'events': events,
            'mean': round(mean, 2) if events > 0 else 'N/A'
        })
    
    # Add total row
    stats.append({
        'range': 'Total',
        'events': total_events,
        'mean': round(filtered_df['invariant_mass'].mean(), 2) if total_events > 0 else 'N/A'
    })
    
    return stats

def create_histogram(df, particle_types, title, num_bins, log_scale, show_total=False):
    """Create a histogram with Plotly"""
    fig = go.Figure()
    
    colors = {
        1: 'blue',
        2: 'red',
        3: 'green',
        '4ee': 'darkblue',
        '4mm': 'darkred',
        '4me': 'purple'
    }
    
    labels = {
        1: 'Electrons',
        2: 'Muons',
        3: 'Photons'
    }
    
    # Use provided number of bins or default to 20
    try:
        bins = int(num_bins) if num_bins and int(num_bins) > 0 else 20
    except (ValueError, TypeError):
        bins = 20
    
    # For dilepton with total, use stacked mode
    if show_total:
        for ptype in particle_types:
            if isinstance(ptype, int):
                filtered_df = df[df['particle_type'] == ptype]
                label = labels.get(ptype, str(ptype))
                color = colors.get(ptype, 'gray')
                
                if len(filtered_df) > 0:
                    mass_values = filtered_df['invariant_mass'].values
                    
                    fig.add_trace(go.Histogram(
                        x=mass_values,
                        name=label,
                        marker_color=color,
                        opacity=0.7,
                        nbinsx=bins
                    ))
        
        # Update to stack mode for dilepton
        fig.update_layout(barmode='stack')
    else:
        # For other plots, use overlay mode
        for ptype in particle_types:
            if isinstance(ptype, int):
                filtered_df = df[df['particle_type'] == ptype]
                label = labels.get(ptype, str(ptype))
                color = colors.get(ptype, 'gray')
            else:
                # Handle string particle types (e.g., '4ee', '4mm', '4me')
                filtered_df = df[df['combination'].str.lower() == ptype.lower()]
                label = ptype.upper()
                color = colors.get(ptype.lower(), 'gray')
            
            if len(filtered_df) > 0:
                mass_values = filtered_df['invariant_mass'].values
                
                fig.add_trace(go.Histogram(
                    x=mass_values,
                    name=label,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=bins
                ))
        
        fig.update_layout(barmode='overlay')
    
    fig.update_layout(
        title=title,
        xaxis_title='Invariant Mass (GeV)',
        yaxis_title='Events',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # FIXED: Only apply log scale to X-axis, not Y-axis
    if log_scale:
        fig.update_xaxes(type='log')
        # Removed: fig.update_yaxes(type='log')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = ['.csv', '.txt']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({'error': 'Only CSV and TXT files are allowed'}), 400
    
    try:
        content = file.read()
        df = parse_csv(content)
        
        if len(df) == 0:
            return jsonify({'error': 'No valid data found in file'}), 400
        
        # Fixed: Changed from bin_width to num_bins
        num_bins = request.form.get('num_bins', 20)
        log_scale = request.form.get('log_scale') == 'true'
        view_mode = request.form.get('view_mode', 'all')
        
        # Create histograms
        plots = {}
        
        if view_mode in ['all', 'dilepton']:
            plots['dilepton'] = create_histogram(
                df, [1, 2], 
                'Dilepton Invariant Mass Distribution', 
                num_bins, log_scale, show_total=True
            )
        
        if view_mode in ['all', 'fourlepton']:
            plots['fourlepton'] = create_histogram(
                df, ['4ee', '4mm', '4me'], 
                'Four Lepton Invariant Mass Distribution', 
                num_bins, log_scale, show_total=False
            )
        
        if view_mode in ['all', 'diphoton']:
            plots['diphoton'] = create_histogram(
                df, [3], 
                'Diphoton Invariant Mass Distribution', 
                num_bins, log_scale, show_total=False
            )
        
        # Calculate statistics
        electron_stats = calculate_statistics(df, [1])
        muon_stats = calculate_statistics(df, [2])
        photon_stats = calculate_statistics(df, [3])
        
        return jsonify({
            'plots': plots,
            'statistics': {
                'electrons': electron_stats,
                'muons': muon_stats,
                'photons': photon_stats
            }
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback_msg}")
        return jsonify({'error': f'Error processing file: {error_msg}'}), 500

if __name__ == '__main__':
    app.run(debug=True)