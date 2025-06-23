import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import seaborn as sns

from test import DeliveryOptimizer, DeliveryPoint, Factory, Truck

# Your original classes here (Factory, DeliveryPoint, Truck, etc.)
# ... [All your original class definitions remain exactly the same] ...

class StreamlitDeliveryApp:
    def __init__(self):
        self.optimizer = None
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if 'optimizer' not in st.session_state:
            st.session_state.optimizer = None
        if 'current_day' not in st.session_state:
            st.session_state.current_day = 1
        if 'month_progress' not in st.session_state:
            st.session_state.month_progress = []
    
    def setup_optimizer(self):
        st.sidebar.header("System Configuration")
        num_trucks = st.sidebar.slider("Number of Trucks", 2, 10, 4)
        truck_capacity = st.sidebar.slider("Truck Capacity (kg)", 1000, 3000, 1500, 100)
        days_in_month = st.sidebar.slider("Days in Month", 20, 31, 30)
        
        if st.sidebar.button("Initialize System"):
            st.session_state.optimizer = DeliveryOptimizer(
                num_trucks=num_trucks,
                truck_capacity=truck_capacity,
                days_in_month=days_in_month
            )
            st.session_state.current_day = 1
            st.session_state.month_progress = []
            st.success("Delivery optimization system initialized!")
    
    def generate_daily_points(self):
        if st.session_state.optimizer:
            min_points = st.sidebar.slider("Minimum Points", 5, 30, 15)
            max_points = st.sidebar.slider("Maximum Points", min_points, 50, 25)
            num_points = random.randint(min_points, max_points)
            return st.session_state.optimizer.generate_delivery_points(num_points)
        return []
    
    def run_day(self):
        if st.session_state.optimizer:
            points = self.generate_daily_points()
            
            st.subheader(f"Day {st.session_state.current_day} Operations")
            
            with st.spinner("Optimizing deliveries..."):
                st.session_state.optimizer.run_daily_operations(points, show_visualization=False)
                
                # Display visualizations
                self.display_daily_visualizations()
                
                # Advance day
                st.session_state.optimizer.advance_day()
                st.session_state.current_day += 1
                
                # Track progress
                progress = {
                    'day': st.session_state.current_day - 1,
                    'circuits': [t.monthly_circuits for t in st.session_state.optimizer.trucks],
                    'distance': [t.monthly_distance for t in st.session_state.optimizer.trucks]
                }
                st.session_state.month_progress.append(progress)
                
                if st.session_state.current_day > st.session_state.optimizer.days_in_month:
                    st.balloons()
                    st.success("Month completed! System will reset for new month.")
                    st.session_state.current_day = 1
    
    def display_daily_visualizations(self):
        if not st.session_state.optimizer:
            return
            
        optimizer = st.session_state.optimizer
        visualizer = optimizer.visualizer
        
        st.subheader("Route Network")
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_route_network_base(ax, optimizer.trucks, optimizer.unassigned_points, optimizer.current_day)
        st.pyplot(fig)
        
        st.subheader("Daily Performance")
        fig = self._create_daily_performance_figure(optimizer.trucks, optimizer.current_day)
        st.pyplot(fig)
    
    def _plot_route_network_base(self, ax, trucks, unassigned_points=None, day=1, show_details=True):
        """Base plotting function for route network"""
        factory = Factory()
        
        # Plot factory
        ax.scatter(factory.x, factory.y, c='red', s=500, marker='s', 
                  edgecolors='black', linewidth=2, label='Factory', zorder=10)
        ax.annotate('FACTORY', (factory.x, factory.y), 
                   xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # Track statistics for summary
        total_points = 0
        total_weight = 0
        total_distance = 0
        
        # Plot each truck's route
        for i, truck in enumerate(trucks):
            if not truck.current_load:
                continue
                
            # Get optimized route
            route_points = self._optimize_route_order(truck.current_load)
            color = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][i % 5]
            
            # Create full route including factory
            x_coords = [factory.x] + [p.x for p in route_points] + [factory.x]
            y_coords = [factory.y] + [p.y for p in route_points] + [factory.y]
            
            # Plot route lines
            ax.plot(x_coords, y_coords, '-', color=color, linewidth=3, alpha=0.8,
                   label=f'Truck {truck.truck_id}')
            
            # Plot delivery points with weight-based sizing
            weights = [p.weight_kg for p in route_points]
            scatter = ax.scatter([p.x for p in route_points], [p.y for p in route_points],
                               s=[w/3 + 50 for w in weights], c=color, 
                               edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
            
            # Add point labels if requested
            if show_details and len(route_points) <= 15:
                for j, point in enumerate(route_points):
                    ax.annotate(f'P{point.id}\n{point.weight_kg}kg', 
                               (point.x, point.y), xytext=(3, 3), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            
            # Update statistics
            total_points += len(route_points)
            total_weight += sum(weights)
            total_distance += truck.daily_distance
        
        # Plot unassigned points
        if unassigned_points:
            ax.scatter([p.x for p in unassigned_points], [p.y for p in unassigned_points],
                      s=[p.weight_kg/3 + 50 for p in unassigned_points], 
                      c='gray', marker='x', linewidths=2, alpha=0.7,
                      label=f'Unassigned ({len(unassigned_points)})')
        
        # Enhance plot appearance
        ax.set_title(f'Delivery Network - Day {day}\n'
                    f'Total: {total_points} points, {total_weight}kg, {total_distance:.1f}km', 
                    fontsize=12)
        ax.set_xlabel('X Coordinate (km)')
        ax.set_ylabel('Y Coordinate (km)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_aspect('equal')
        ax.margins(0.1)
    
    def _optimize_route_order(self, points: List[DeliveryPoint]) -> List[DeliveryPoint]:
        """Apply nearest neighbor algorithm to optimize delivery order"""
        if not points:
            return []
            
        unvisited = points.copy()
        route = []
        current_x, current_y = 0.0, 0.0
        
        while unvisited:
            nearest = min(unvisited,
                         key=lambda p: math.sqrt((p.x-current_x)**2 + (p.y-current_y)**2))
            route.append(nearest)
            current_x, current_y = nearest.x, nearest.y
            unvisited.remove(nearest)
        
        return route
    
    def _create_daily_performance_figure(self, trucks: List[Truck], day: int):
        """Create daily performance figure for Streamlit"""
        active_trucks = [t for t in trucks if t.current_load or t.daily_circuits > 0]
        if not active_trucks:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        truck_ids = [t.truck_id for t in active_trucks]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(active_trucks)]
        
        # Daily circuits
        circuits = [t.daily_circuits for t in active_trucks]
        bars1 = ax1.bar(truck_ids, circuits, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Daily Circuits Completed')
        ax1.set_ylabel('Number of Circuits')
        ax1.set_xlabel('Truck ID')
        for bar, circuit in zip(bars1, circuits):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{circuit}', ha='center', va='bottom', fontweight='bold')
        
        # Daily distance
        distances = [t.daily_distance for t in active_trucks]
        bars2 = ax2.bar(truck_ids, distances, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Daily Distance Traveled')
        ax2.set_ylabel('Distance (km)')
        ax2.set_xlabel('Truck ID')
        for bar, dist in zip(bars2, distances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{dist:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Current load weight
        weights = [t.current_weight for t in active_trucks]
        bars3 = ax3.bar(truck_ids, weights, color=colors, alpha=0.8, edgecolor='black')
        ax3.axhline(y=1500, color='red', linestyle='--', linewidth=2, label='Capacity Limit')
        ax3.set_title('Current Load Weight')
        ax3.set_ylabel('Weight (kg)')
        ax3.set_xlabel('Truck ID')
        ax3.legend()
        for bar, weight in zip(bars3, weights):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{weight}kg', ha='center', va='bottom', fontweight='bold')
        
        # Load utilization percentage
        utilization = [(w/1500)*100 for w in weights]
        bars4 = ax4.bar(truck_ids, utilization, color=colors, alpha=0.8, edgecolor='black')
        ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, label='100% Capacity')
        ax4.set_title('Load Utilization')
        ax4.set_ylabel('Utilization (%)')
        ax4.set_xlabel('Truck ID')
        ax4.legend()
        for bar, util in zip(bars4, utilization):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Daily Performance Summary - Day {day}', fontsize=14)
        return fig
    
    def display_monthly_summary(self):
        if not st.session_state.optimizer or not st.session_state.month_progress:
            return
            
        optimizer = st.session_state.optimizer
        visualizer = optimizer.visualizer
        
        st.subheader("Monthly Summary")
        fig = self._create_monthly_summary_figure(optimizer.trucks, optimizer.current_day - 1)
        st.pyplot(fig)
    
    def _create_monthly_summary_figure(self, trucks: List[Truck], current_day: int):
        """Create monthly summary figure for Streamlit"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create custom grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        truck_ids = [t.truck_id for t in trucks]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(trucks)]
        
        # 1. Monthly circuits (spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        circuits = [t.monthly_circuits for t in trucks]
        bars1 = ax1.bar(truck_ids, circuits, color=colors, alpha=0.8, edgecolor='black')
        ax1.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Minimum Required (20)')
        ax1.set_title('Monthly Circuits Progress', fontweight='bold')
        ax1.set_ylabel('Number of Circuits')
        ax1.set_xlabel('Truck ID')
        ax1.legend()
        for bar, circuit in zip(bars1, circuits):
            height = bar.get_height()
            status = "✓" if circuit >= 20 else "⚠"
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{circuit} {status}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Monthly distance (spanning 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        distances = [t.monthly_distance for t in trucks]
        bars2 = ax2.bar(truck_ids, distances, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Monthly Distance Traveled', fontweight='bold')
        ax2.set_ylabel('Distance (km)')
        ax2.set_xlabel('Truck ID')
        for bar, dist in zip(bars2, distances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{dist:.1f}km', ha='center', va='bottom', fontweight='bold')
        
        # 3. Circuit completion rate pie chart
        ax3 = fig.add_subplot(gs[1, 0])
        completed = sum(1 for t in trucks if t.monthly_circuits >= 20)
        behind = len(trucks) - completed
        if behind > 0:
            ax3.pie([completed, behind], labels=['On Track', 'Behind'], 
                   colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90)
        else:
            ax3.pie([completed], labels=['All On Track'], 
                   colors=['#2ECC71'], autopct='%1.1f%%', startangle=90)
        ax3.set_title('Circuit Completion Status', fontweight='bold')
        
        # 4. Distance distribution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(distances, bins=max(3, len(trucks)//2), color='skyblue', 
                alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances):.1f}km')
        ax4.set_title('Distance Distribution', fontweight='bold')
        ax4.set_xlabel('Distance (km)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Workload balance
        ax5 = fig.add_subplot(gs[1, 2])
        workload_ratio = [d/max(distances) if max(distances) > 0 else 0 for d in distances]
        bars5 = ax5.bar(truck_ids, workload_ratio, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_title('Workload Balance', fontweight='bold')
        ax5.set_ylabel('Relative Workload')
        ax5.set_xlabel('Truck ID')
        ax5.set_ylim(0, 1.1)
        
        # 6. Progress timeline (simulated)
        ax6 = fig.add_subplot(gs[1, 3])
        days = list(range(1, current_day + 1))
        for i, truck in enumerate(trucks):
            # Simulate progress over time
            progress = [(truck.monthly_circuits / current_day) * day for day in days]
            ax6.plot(days, progress, color=colors[i], linewidth=2, 
                    marker='o', markersize=4, label=f'Truck {truck.truck_id}')
        ax6.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Target (20)')
        ax6.set_title('Circuit Progress Timeline', fontweight='bold')
        ax6.set_xlabel('Days')
        ax6.set_ylabel('Circuits Completed')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary statistics table (spanning full width)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary table
        table_data = []
        for truck in trucks:
            remaining_circuits = max(0, 20 - truck.monthly_circuits)
            avg_distance_per_circuit = truck.monthly_distance / max(1, truck.monthly_circuits)
            status = "✓ On Track" if truck.monthly_circuits >= 20 else f"⚠ Need {remaining_circuits} more"
            
            table_data.append([
                f'Truck {truck.truck_id}',
                f'{truck.monthly_circuits}/20',
                f'{truck.monthly_distance:.1f} km',
                f'{avg_distance_per_circuit:.1f} km',
                status
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Truck', 'Circuits', 'Total Distance', 'Avg Distance/Circuit', 'Status'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the status column
        for i in range(len(trucks)):
            if trucks[i].monthly_circuits >= 20:
                table[(i+1, 4)].set_facecolor('#D5F4E6')  # Light green
            else:
                table[(i+1, 4)].set_facecolor('#FADBD8')  # Light red
        
        plt.suptitle(f'Monthly Performance Dashboard - Day {current_day}', 
                    fontsize=16, fontweight='bold')
        return fig
    
    def run(self):
        st.set_page_config(layout="wide", page_title="Delivery Optimization System")
        st.title("🚛 Delivery Optimization System")
        
        self.setup_optimizer()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["Daily Operations", "Monthly Summary", "System Configuration"])
        
        with tab1:
            st.header("Daily Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Run Daily Simulation"):
                    self.run_day()
            
            with col2:
                if st.button("Fast Forward Month"):
                    for _ in range(st.session_state.optimizer.days_in_month - st.session_state.current_day + 1):
                        points = self.generate_daily_points()
                        st.session_state.optimizer.run_daily_operations(points, show_visualization=False)
                        st.session_state.optimizer.advance_day()
                        st.session_state.current_day += 1
                    st.rerun()
            
            if st.session_state.optimizer:
                # Display current day
                st.subheader(f"Day {st.session_state.current_day}/{st.session_state.optimizer.days_in_month}")

                # Truck stats
                st.subheader("Truck Statistics")
                truck_data = []
                for truck in st.session_state.optimizer.trucks:
                    truck_data.append({
                        "Truck ID": truck.truck_id,
                        "Daily Circuits": truck.daily_circuits,
                        "Monthly Circuits": f"{truck.monthly_circuits}/20",
                        "Current Load (kg)": f"{truck.current_weight}/{truck.capacity}",
                        "Daily Distance (km)": round(truck.daily_distance, 2),
                        "Monthly Distance (km)": round(truck.monthly_distance, 2)
                    })
                st.dataframe(pd.DataFrame(truck_data), use_container_width=True)

                # Delivery points
                st.subheader("Delivery Points")
                if st.session_state.optimizer.unassigned_points:
                    points_data = []
                    for point in st.session_state.optimizer.unassigned_points:
                        points_data.append({
                            "ID": point.id,
                            "X": point.x,
                            "Y": point.y,
                            "Weight (kg)": point.weight_kg,
                            "Distance from Factory": round(point.distance_from_factory, 2)
                        })
                    st.dataframe(pd.DataFrame(points_data), use_container_width=True)
        
        with tab2:
            self.display_monthly_summary()
        
        with tab3:
            st.header("System Configuration")
            st.write("Current system parameters:")
            if st.session_state.optimizer:
                st.json({
                    "Number of Trucks": len(st.session_state.optimizer.trucks),
                    "Truck Capacity (kg)": st.session_state.optimizer.trucks[0].capacity,
                    "Days in Month": st.session_state.optimizer.days_in_month
                })

# Run the app
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    app = StreamlitDeliveryApp()
    app.run()
