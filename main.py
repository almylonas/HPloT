from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import numpy as np
import json
import io
import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URL'] = os.environ.get('DATABASE_URL', 'sqlite:///particle_data.db')
# Fix for Railway PostgreSQL URLs
if app.config['SQLALCHEMY_DATABASE_URL'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URL'] = app.config['SQLALCHEMY_DATABASE_URL'].replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Energy ranges for analysis
ENERGY_RANGES = {
    'R1': (2, 4),
    'R2': (7, 13),
    'R3': (80, 100),
    'R4': (900, 1100),
    'R5': (1400, 1600)
}

# Database Models
class Group(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    submissions = db.relationship('Submission', backref='group', lazy=True, cascade='all, delete-orphan')

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    data_entries = db.relationship('DataEntry', backref='submission', lazy=True, cascade='all, delete-orphan')

class DataEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    invariant_mass = db.Column(db.Float, nullable=False)
    particle_type = db.Column(db.Integer, nullable=False)
    combination = db.Column(db.String(10), default='')

# Create tables
with app.app_context():
    db.create_all()

def parse_csv(file_content):
    """Parse the CSV content and return a DataFrame"""
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
    
    df = df.dropna(subset=['invariant_mass', 'particle_type'])
    df['invariant_mass'] = pd.to_numeric(df['invariant_mass'], errors='coerce')
    df['particle_type'] = pd.to_numeric(df['particle_type'], errors='coerce')
    df = df.dropna(subset=['invariant_mass', 'particle_type'])
    df['particle_type'] = df['particle_type'].astype(int)
    df['combination'] = df['combination'].fillna('').astype(str).str.strip()
    
    return df

def get_dataframe_from_db(group_id=None):
    """Get DataFrame from database entries"""
    query = DataEntry.query
    
    if group_id:
        query = query.join(Submission).filter(Submission.group_id == group_id)
    
    entries = query.all()
    
    data = {
        'invariant_mass': [e.invariant_mass for e in entries],
        'particle_type': [e.particle_type for e in entries],
        'combination': [e.combination for e in entries]
    }
    
    return pd.DataFrame(data)

def calculate_statistics(df, particle_types):
    """Calculate statistics for energy ranges"""
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
    
    bins = num_bins if num_bins and num_bins > 0 else 30
    
    if show_total:
        for ptype in particle_types:
            if isinstance(ptype, int):
                filtered_df = df[df['particle_type'] == ptype]
                
                if log_scale:
                    filtered_df = filtered_df[filtered_df['invariant_mass'] > 0]
                
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
        
        fig.update_layout(barmode='stack')
    else:
        for ptype in particle_types:
            if isinstance(ptype, int):
                filtered_df = df[df['particle_type'] == ptype]
                label = labels.get(ptype, str(ptype))
                color = colors.get(ptype, 'gray')
            else:
                filtered_df = df[df['combination'].str.lower() == ptype.lower()]
                label = ptype.upper()
                color = colors.get(ptype.lower(), 'gray')
            
            if log_scale:
                filtered_df = filtered_df[filtered_df['invariant_mass'] > 0]
            
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
    
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
        if len(df[df['invariant_mass'] <= 0]) > 0:
            fig.update_layout(
                title=title + " (non-positive values excluded for log scale)"
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/groups', methods=['GET'])
def get_groups():
    """Get all groups"""
    groups = Group.query.order_by(Group.created_at.desc()).all()
    return jsonify([{
        'id': g.id,
        'name': g.name,
        'submission_count': len(g.submissions),
        'created_at': g.created_at.isoformat()
    } for g in groups])

@app.route('/api/groups', methods=['POST'])
def create_group():
    """Create a new group"""
    data = request.json
    group_name = data.get('name', '').strip()
    
    if not group_name:
        return jsonify({'error': 'Group name is required'}), 400
    
    if Group.query.filter_by(name=group_name).first():
        return jsonify({'error': 'Group name already exists'}), 400
    
    group = Group(name=group_name)
    db.session.add(group)
    db.session.commit()
    
    return jsonify({
        'id': group.id,
        'name': group.name,
        'created_at': group.created_at.isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload():
    """Upload data to a group"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    group_id = request.form.get('group_id')
    if not group_id:
        return jsonify({'error': 'Group selection is required'}), 400
    
    group = Group.query.get(group_id)
    if not group:
        return jsonify({'error': 'Group not found'}), 404
    
    allowed_extensions = ['.csv', '.txt']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({'error': 'Only CSV and TXT files are allowed'}), 400
    
    try:
        content = file.read()
        df = parse_csv(content)
        
        if len(df) == 0:
            return jsonify({'error': 'No valid data found in file'}), 400
        
        # Create submission
        submission = Submission(
            group_id=group_id,
            filename=file.filename
        )
        db.session.add(submission)
        db.session.flush()
        
        # Add data entries
        for _, row in df.iterrows():
            entry = DataEntry(
                submission_id=submission.id,
                invariant_mass=float(row['invariant_mass']),
                particle_type=int(row['particle_type']),
                combination=str(row['combination'])
            )
            db.session.add(entry)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Data submitted successfully',
            'submission_id': submission.id,
            'entries_count': len(df)
        })
    
    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze data for a group or all groups"""
    try:
        data = request.json
        group_id = data.get('group_id')
        num_bins = data.get('num_bins', 30)
        log_scale = data.get('log_scale', False)
        view_mode = data.get('view_mode', 'all')
        
        # Get data from database
        df = get_dataframe_from_db(group_id)
        
        if len(df) == 0:
            return jsonify({'error': 'No data available'}), 400
        
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
            },
            'total_entries': len(df)
        })
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Error analyzing data: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)