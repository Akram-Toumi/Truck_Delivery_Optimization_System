import numpy as np
import matplotlib.pyplot as plt
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import seaborn as sns

@dataclass
class DeliveryPoint:
    """Represents a single delivery point with coordinates and weight requirement"""
    id: int
    x: float
    y: float
    weight_kg: int
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def distance_from_factory(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def distance_to(self, other_point: 'DeliveryPoint') -> float:
        """Calculate distance to another delivery point"""
        return math.sqrt((self.x - other_point.x)**2 + (self.y - other_point.y)**2)
    
    def __str__(self) -> str:
        return f"Point {self.id}: ({self.x:.2f}, {self.y:.2f}), {self.weight_kg}kg"

class Factory:
    """Represents the central factory/depot"""
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to_point(self, point: DeliveryPoint) -> float:
        """Calculate distance from factory to a delivery point"""
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)
    
    def __str__(self) -> str:
        return f"Factory at ({self.x}, {self.y})"

class Truck:
    def __init__(self, truck_id: int, capacity: int = 1500):
        self.truck_id = truck_id
        self.capacity = capacity
        self.current_load: List[DeliveryPoint] = []
        self.daily_circuits = 0
        self.monthly_circuits = 0
        self.total_distance = 0.0
        self.monthly_distance = 0.0
        self.daily_distance = 0.0

    @property
    def current_weight(self) -> int:
        return sum(point.weight_kg for point in self.current_load)

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.current_weight

    def can_add_point(self, point: DeliveryPoint) -> bool:
        return self.current_weight + point.weight_kg <= self.capacity

    def add_point(self, point: DeliveryPoint) -> bool:
        if self.can_add_point(point):
            self.current_load.append(point)
            return True
        return False

    def reset_daily_stats(self):
        self.daily_circuits = 0
        self.daily_distance = 0.0
        self.current_load = []

    def complete_circuit(self, distance: float):
        self.daily_circuits += 1
        self.monthly_circuits += 1
        self.total_distance += distance
        self.monthly_distance += distance
        self.daily_distance += distance

    def __str__(self):
        return (f"Truck {self.truck_id}: {self.current_weight}/{self.capacity}kg | "
                f"Daily: {self.daily_circuits} circuits, {self.daily_distance:.1f}km | "
                f"Monthly: {self.monthly_circuits}/20 circuits, {self.monthly_distance:.1f}km")

class MultinomialAssignmentStrategy:
    def __init__(self, num_trucks: int):
        self.num_trucks = num_trucks
        self.balance_factor = 0.6
        self.circuit_weight = 0.3
        self.distance_weight = 0.7

    def get_distance_probabilities(self, trucks: List[Truck]) -> np.ndarray:
        """Calculate probabilities based on monthly distance (lower distance = higher probability)"""
        if not any(t.monthly_distance for t in trucks):
            return np.ones(len(trucks))/len(trucks)
        
        distances = np.array([t.monthly_distance for t in trucks])
        inv_distances = 1/(1 + distances)
        return inv_distances / np.sum(inv_distances)

    def get_circuit_probabilities(self, trucks: List[Truck]) -> np.ndarray:
        """Calculate probabilities based on circuit counts (fewer circuits = higher probability)"""
        remaining = np.array([max(0, 20 - t.monthly_circuits) for t in trucks])
        if not any(remaining):
            return np.ones(len(trucks))/len(trucks)
        return remaining / np.sum(remaining)

    def get_combined_probabilities(self, trucks: List[Truck]) -> np.ndarray:
        """Combine distance and circuit probabilities"""
        p_dist = self.get_distance_probabilities(trucks)
        p_circ = self.get_circuit_probabilities(trucks)
        return (self.distance_weight*p_dist + self.circuit_weight*p_circ)

    def optimize_assignment(self, points: List[DeliveryPoint], 
                          trucks: List[Truck],
                          max_iterations: int = 500) -> Tuple[Dict, List[DeliveryPoint]]:
        probabilities = self.get_combined_probabilities(trucks)
        best_solution = None
        best_score = float('-inf')
        
        for _ in range(max_iterations):
            temp_trucks = [Truck(t.truck_id, t.capacity) for t in trucks]
            temp_unassigned = []
            point_order = random.sample(points, len(points))
            
            for point in point_order:
                truck_idx = np.random.choice(len(temp_trucks), p=probabilities)
                if temp_trucks[truck_idx].can_add_point(point):
                    temp_trucks[truck_idx].add_point(point)
                else:
                    temp_unassigned.append(point)
            
            solution_valid = True
            truck_stats = []
            total_distance = 0
            
            for truck in temp_trucks:
                if truck.current_load:
                    route_distance = self._calculate_optimized_route(truck.current_load)
                    if truck.current_weight > truck.capacity:
                        solution_valid = False
                        break
                    
                    truck_stats.append({
                        'load': truck.current_weight,
                        'distance': route_distance,
                        'circuits_needed': math.ceil(truck.current_weight/truck.capacity)
                    })
                    total_distance += route_distance
            
            if solution_valid and truck_stats:
                current_score = self._calculate_solution_score(truck_stats)
                if current_score > best_score:
                    best_score = current_score
                    best_solution = {
                        'truck_assignments': [t.current_load.copy() for t in temp_trucks],
                        'truck_distances': [next((s['distance'] for s in truck_stats if s['load'] == t.current_weight), 0.0) for t in temp_trucks],
                        'total_distance': total_distance,
                        'score': current_score,
                        'unassigned_points': temp_unassigned.copy()
                    }
        
        if best_solution:
            return best_solution, best_solution['unassigned_points']
        return {}, points

    def _calculate_optimized_route(self, points: List[DeliveryPoint]) -> float:
        """Calculate optimized route distance using nearest neighbor"""
        if not points:
            return 0.0
            
        unvisited = points.copy()
        current_x, current_y = 0.0, 0.0
        total_distance = 0.0
        
        while unvisited:
            nearest = min(unvisited,
                         key=lambda p: math.sqrt((p.x-current_x)**2 + (p.y-current_y)**2))
            dist = math.sqrt((nearest.x-current_x)**2 + (nearest.y-current_y)**2)
            total_distance += dist
            current_x, current_y = nearest.x, nearest.y
            unvisited.remove(nearest)
        
        # Return to factory
        total_distance += math.sqrt(current_x**2 + current_y**2)
        return total_distance

    def _calculate_solution_score(self, truck_stats: List[Dict]) -> float:
        """Calculate overall solution quality score"""
        loads = [t['load'] for t in truck_stats]
        distances = [t['distance'] for t in truck_stats]
        
        load_utilization = np.mean(loads) / 1500
        distance_balance = 1 - (np.std(distances)/np.mean(distances)) if np.mean(distances) > 0 else 1
        
        return (self.balance_factor * load_utilization + 
               (1 - self.balance_factor) * distance_balance)

class DeliveryVisualizer:
    def __init__(self, factory: Factory):
        self.factory = factory
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        plt.style.use('default')
        
    def plot_route_network(self, trucks: List[Truck], unassigned_points: List[DeliveryPoint] = None,
                          day: int = 1, show_details: bool = True):
        """Enhanced route visualization with better colors and annotations"""
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot factory with enhanced styling
        ax.scatter(self.factory.x, self.factory.y, c='red', s=500, marker='s', 
                  edgecolors='black', linewidth=2, label='Factory', zorder=10)
        ax.annotate('FACTORY', (self.factory.x, self.factory.y), 
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
            color = self.colors[i % len(self.colors)]
            
            # Create full route including factory
            x_coords = [self.factory.x] + [p.x for p in route_points] + [self.factory.x]
            y_coords = [self.factory.y] + [p.y for p in route_points] + [self.factory.y]
            
            # Plot route lines
            ax.plot(x_coords, y_coords, '-', color=color, linewidth=3, alpha=0.8,
                   label=f'Truck {truck.truck_id}')
            
            # Plot delivery points with weight-based sizing
            weights = [p.weight_kg for p in route_points]
            scatter = ax.scatter([p.x for p in route_points], [p.y for p in route_points],
                               s=[w/3 + 50 for w in weights], c=color, 
                               edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
            
            # Add point labels if requested
            if show_details and len(route_points) <= 15:  # Only show labels for reasonable number of points
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
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Set equal aspect ratio and add margins
        ax.set_aspect('equal')
        ax.margins(0.1)
        
        plt.show()
    
    def _optimize_route_order(self, points: List[DeliveryPoint]) -> List[DeliveryPoint]:
        """Apply nearest neighbor algorithm to optimize delivery order"""
        if not points:
            return []
            
        unvisited = points.copy()
        route = []
        current_x, current_y = self.factory.x, self.factory.y
        
        while unvisited:
            nearest = min(unvisited,
                         key=lambda p: math.sqrt((p.x-current_x)**2 + (p.y-current_y)**2))
            route.append(nearest)
            current_x, current_y = nearest.x, nearest.y
            unvisited.remove(nearest)
        
        return route
    
    def plot_daily_performance(self, trucks: List[Truck], day: int):
        """Enhanced daily performance visualization"""
        active_trucks = [t for t in trucks if t.current_load or t.daily_circuits > 0]
        if not active_trucks:
            print("No active trucks to display")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        truck_ids = [t.truck_id for t in active_trucks]
        colors = [self.colors[i % len(self.colors)] for i in range(len(active_trucks))]
        
        # Daily circuits
        circuits = [t.daily_circuits for t in active_trucks]
        bars1 = ax1.bar(truck_ids, circuits, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Daily Circuits Completed', fontweight='bold')
        ax1.set_ylabel('Number of Circuits')
        ax1.set_xlabel('Truck ID')
        for bar, circuit in zip(bars1, circuits):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{circuit}', ha='center', va='bottom', fontweight='bold')
        
        # Daily distance
        distances = [t.daily_distance for t in active_trucks]
        bars2 = ax2.bar(truck_ids, distances, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Daily Distance Traveled', fontweight='bold')
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
        ax3.set_title('Current Load Weight', fontweight='bold')
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
        ax4.set_title('Load Utilization', fontweight='bold')
        ax4.set_ylabel('Utilization (%)')
        ax4.set_xlabel('Truck ID')
        ax4.legend()
        for bar, util in zip(bars4, utilization):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Daily Performance Summary - Day {day}', fontsize=16, fontweight='bold')
        plt.show()
    
    def plot_monthly_summary(self, trucks: List[Truck], current_day: int):
        """Comprehensive monthly summary with multiple visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create custom grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        truck_ids = [t.truck_id for t in trucks]
        colors = [self.colors[i % len(self.colors)] for i in range(len(trucks))]
        
        # 1. Monthly circuits (spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        circuits = [t.monthly_circuits for t in trucks]
        bars1 = ax1.bar(truck_ids, circuits, color=colors, alpha=0.8, edgecolor='black')
        ax1.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Minimum Required (20)')
        ax1.set_title('Monthly Circuits Progress', fontweight='bold', fontsize=14)
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
        ax2.set_title('Monthly Distance Traveled', fontweight='bold', fontsize=14)
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
        
        plt.suptitle(f'Monthly Performance Dashboard - Day {current_day}/30', 
                    fontsize=18, fontweight='bold')
        plt.show()

class DeliveryOptimizer:
    def __init__(self, num_trucks: int = 4, truck_capacity: int = 1500, days_in_month: int = 30):
        self.trucks = [Truck(i+1, truck_capacity) for i in range(num_trucks)]
        self.strategy = MultinomialAssignmentStrategy(num_trucks)
        self.factory = Factory()
        self.visualizer = DeliveryVisualizer(factory=self.factory)
        self.days_in_month = days_in_month
        self.current_day = 1
        self.unassigned_points = []
    
    def generate_delivery_points(self, num_points: int = 20) -> List[DeliveryPoint]:
        """Generate random delivery points"""
        points = []
        for i in range(num_points):
            angle = random.uniform(0, 2*np.pi)
            distance = random.uniform(5, 25)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            weight = random.randint(50, 800)
            points.append(DeliveryPoint(i+1, x, y, weight))
        return points
    
    def run_daily_operations(self, points: List[DeliveryPoint], show_visualization: bool = True):
        """Run full daily optimization with enhanced visualization"""
        all_points = self.unassigned_points + points
        self.unassigned_points = []
        
        solution, self.unassigned_points = self.strategy.optimize_assignment(
            all_points, self.trucks
        )
        
        if solution:
            for i, truck in enumerate(self.trucks):
                truck.current_load = solution['truck_assignments'][i]
                if truck.current_load:
                    distance = solution['truck_distances'][i]
                    truck.complete_circuit(distance)
        
        # Show visualizations
        if show_visualization:
            self.visualizer.plot_route_network(self.trucks, self.unassigned_points, 
                                             day=self.current_day, show_details=True)
            self.visualizer.plot_daily_performance(self.trucks, self.current_day)
    
    def advance_day(self):
        """Move to next day and reset daily stats"""
        for truck in self.trucks:
            truck.reset_daily_stats()
        
        self.current_day += 1
        if self.current_day > self.days_in_month:
            self._start_new_month()
    
    def _start_new_month(self):
        """Reset monthly statistics"""
        self.current_day = 1
        for truck in self.trucks:
            truck.monthly_circuits = 0
            truck.monthly_distance = 0.0
        print("\n=== NEW MONTH STARTED ===")
    
    def check_monthly_requirements(self):
        """Check if trucks are meeting monthly requirements"""
        print("\nMonthly Requirement Check:")
        for truck in self.trucks:
            status = "✓ OK" if truck.monthly_circuits >= 20 else "⚠ WARNING"
            remaining = max(0, 20 - truck.monthly_circuits)
            print(f"Truck {truck.truck_id}: {truck.monthly_circuits}/20 circuits ({status})" + 
                  (f" - Need {remaining} more" if remaining > 0 else ""))

# Example usage
def demo_system():
    """Demonstrate the enhanced delivery optimization system"""
    print("=== DELIVERY OPTIMIZATION SYSTEM DEMO ===\n")
    
    # Initialize system
    optimizer = DeliveryOptimizer(num_trucks=4, days_in_month=30)
    
    # Run a few days with visualization
    for day in range(1, 4):  # Run first 3 days
        print(f"\n=== DAY {day} ===")
        points = optimizer.generate_delivery_points(random.randint(15, 25))
        optimizer.run_daily_operations(points, show_visualization=True)
        optimizer.advance_day()
    
    # Show monthly summary
    optimizer.visualizer.plot_monthly_summary(optimizer.trucks, optimizer.current_day - 1)
    optimizer.check_monthly_requirements()

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    demo_system()
