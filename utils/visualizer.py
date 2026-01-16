"""
Visualization Module
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

class Visualizer:
    """Class untuk visualisasi data"""
    
    def create_risk_gauge(self, risk_score):
        """Create gauge chart untuk risk score"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_features_table(self, features):
        """Create table dari features"""
        # Convert features dict to dataframe
        features_list = []
        
        for key, value in features.items():
            if isinstance(value, list):
                display_value = ", ".join(str(v) for v in value[:3])
                if len(value) > 3:
                    display_value += f"... (+{len(value)-3} more)"
            elif isinstance(value, bool):
                display_value = "✅" if value else "❌"
            else:
                display_value = str(value)
            
            features_list.append({
                'Feature': key.replace('_', ' ').title(),
                'Value': display_value
            })
        
        return pd.DataFrame(features_list)
    
    def create_history_chart(self, history_data):
        """Create chart dari scan history"""
        if not history_data:
            return None
        
        df = pd.DataFrame(history_data)
        
        fig = px.line(df, x='timestamp', y='risk_score',
                     title='Risk Score History',
                     labels={'risk_score': 'Risk Score', 'timestamp': 'Time'})
        
        fig.update_layout(height=400)
        return fig
    
    def create_pie_chart(self, risk_distribution):
        """Create pie chart untuk risk distribution"""
        labels = list(risk_distribution.keys())
        values = list(risk_distribution.values())
        
        colors = ['green', 'yellow', 'orange', 'red']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker_colors=colors[:len(labels)]
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            height=400
        )
        
        return fig