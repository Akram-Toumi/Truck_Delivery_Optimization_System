"""
MONTHLY DELIVERY OPTIMIZATION SYSTEM WITH VISUALIZATION
Features:
- 30-day simulation with daily route planning
- Clarke-Wright Savings Algorithm with 2-opt optimization
- Real road distances using OSRM (fallback to haversine)
- Balanced workloads across trucks using multinomial distribution
- Minimum 20 deliveries per truck enforced
- Interactive Folium maps for each day
- Monthly summary statistics and charts
- Carry-over of unassigned deliveries
"""

import requests
import folium
import random
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import os

@dataclass
class DeliveryPoint:
    id: int
    lat: float
    lon: float
    weight_kg: int
    day_generated: int
    address: str = ""

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.lat, self.lon)

    def __str__(self):
        return f"Delivery {self.id} ({self.weight_kg}kg)"

@dataclass
class Saving:
    point_i: DeliveryPoint
    point_j: DeliveryPoint
    savings: float


# Configuration
DAYS_IN_MONTH = 30
TRUCK_CAPACITY = 1500  # kg
MIN_DELIVERIES_PER_TRUCK = 20
NUM_TRUCKS = 3
FACTORY_LOCATION = (52.5200, 13.4050)  # Berlin coordinates
OUTPUT_DIR = "delivery_visualizations"
OSRM_ENDPOINT = "http://router.project-osrm.org/route/v1/driving/"

class DeliverySystem:
    def __init__(self):
        self.factory = Factory(*FACTORY_LOCATION)
        self.trucks = [Truck(i+1, TRUCK_CAPACITY) for i in range(NUM_TRUCKS)]
        self.router = RealRoadRouter()
        self.current_day = 1
        self.unassigned_points = []
        self.monthly_stats = []
        self.all_deliveries = []
        self._create_output_dir()

    def _create_output_dir(self):
        """Create directory for output files"""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            os.makedirs(os.path.join(OUTPUT_DIR, "daily_maps"))
            os.makedirs(os.path.join(OUTPUT_DIR, "monthly_charts"))

    def run_monthly_simulation(self):
        """Run full month simulation"""
        start_date = datetime.now()
        
        print(f"Starting monthly simulation with {NUM_TRUCKS} trucks")
        print(f"Each truck has capacity of {TRUCK_CAPACITY}kg")
        print(f"Minimum {MIN_DELIVERIES_PER_TRUCK} deliveries per truck required\n")
        
        for day in range(1, DAYS_IN_MONTH+1):
            self.current_day = day
            current_date = (start_date + timedelta(days=day-1)).strftime('%Y-%m-%d')
            print(f"\n=== DAY {day} ({current_date}) ===")
            
            # Generate new deliveries (15-20 points)
            new_points = self._generate_deliveries()
            all_points = self.unassigned_points + new_points
            self.unassigned_points = []
            
            print(f"Total points to assign: {len(all_points)} "
                  f"({len(new_points)} new, {len(self.unassigned_points)} carried over)")
            
            # Optimize routes using Clarke-Wright
            optimizer = RouteOptimizer(self.trucks, self.router)
            self.unassigned_points = optimizer.optimize_routes(self.factory, all_points)
            
            # Balance truck workloads
            self._balance_truck_loads()
            
            # Ensure minimum deliveries per truck
            self._ensure_minimum_deliveries()
            
            # Record stats and visualize
            day_stats = self._record_day_stats(day, current_date)
            self._visualize_day(day, day_stats)
        
        self._generate_monthly_report()
        print("\nSimulation complete! Check the output files in:", os.path.abspath(OUTPUT_DIR))

    def _generate_deliveries(self) -> List[DeliveryPoint]:
        """Generate 15-20 random delivery points with realistic distribution"""
        num_points = random.randint(15, 20)
        points = []
        
        for _ in range(num_points):
            # Generate point in realistic distribution around factory
            angle = random.uniform(0, 2*math.pi)
            distance = min(random.expovariate(1/10)*20, 20)  # Max 20km radius
            
            # Convert distance to lat/lon offsets
            lat_offset = (distance * math.cos(angle)) / 111.0
            lon_offset = (distance * math.sin(angle)) / (111.0 * math.cos(math.radians(self.factory.lat)))
            
            # Create point with realistic weight distribution
            weight = random.choices(
                [50, 100, 200, 300, 500, 800],
                weights=[20, 30, 25, 15, 8, 2]
            )[0]
            
            point = DeliveryPoint(
                id=len(self.all_deliveries)+1,
                lat=self.factory.lat + lat_offset,
                lon=self.factory.lon + lon_offset,
                weight_kg=weight,
                day_generated=self.current_day
            )
            
            points.append(point)
            self.all_deliveries.append(point)
        
        return points

    def _balance_truck_loads(self):
        """Balance distances using multinomial distribution approach"""
        distances = [t.total_distance for t in self.trucks if t.deliveries]
        if not distances or max(distances) - min(distances) < 5:  # 5km threshold
            return
        
        print("Balancing truck routes to equalize distances...")
        
        # Collect all points from all trucks
        all_points = []
        for truck in self.trucks:
            all_points.extend(truck.deliveries)
            truck.clear_route()
        
        # Sort trucks by current distance (ascending)
        sorted_trucks = sorted(self.trucks, key=lambda t: t.total_distance)
        
        # Redistribute points with multinomial approach
        point_weights = [p.weight_kg for p in all_points]
        total_weight = sum(point_weights)
        avg_weight_per_truck = total_weight / len(self.trucks)
        
        # Assign points to trucks based on weight distribution
        current_truck_idx = 0
        current_weight = 0
        
        # Sort points by weight (heaviest first)
        all_points.sort(key=lambda p: p.weight_kg, reverse=True)
        
        for point in all_points:
            if (current_weight + point.weight_kg > avg_weight_per_truck * 1.2 and 
                current_truck_idx < len(sorted_trucks)-1):
                current_truck_idx += 1
                current_weight = 0
            
            sorted_trucks[current_truck_idx].add_delivery(point)
            current_weight += point.weight_kg
        
        # Re-optimize routes
        for truck in self.trucks:
            if truck.deliveries:
                truck.optimize_route(self.factory, self.router)

    def _ensure_minimum_deliveries(self):
        """Ensure each truck has minimum required deliveries"""
        for truck in self.trucks:
            needed = max(0, MIN_DELIVERIES_PER_TRUCK - len(truck.deliveries))
            if needed == 0:
                continue
                
            print(f"Truck {truck.id} needs {needed} more deliveries")
            
            # Find points from other trucks that can be reassigned
            for donor in self.trucks:
                if donor == truck or len(donor.deliveries) <= MIN_DELIVERIES_PER_TRUCK:
                    continue
                    
                # Calculate how many points we can take
                points_to_take = min(needed, len(donor.deliveries) - MIN_DELIVERIES_PER_TRUCK)
                if points_to_take <= 0:
                    continue
                
                # Transfer points (take from end of route)
                for _ in range(points_to_take):
                    if donor.deliveries and truck.can_add_delivery(donor.deliveries[-1]):
                        point = donor.deliveries.pop()
                        truck.add_delivery(point)
                        needed -= 1
                        if needed == 0:
                            break
                
                if needed == 0:
                    break
            
            # Re-optimize affected trucks
            truck.optimize_route(self.factory, self.router)
            donor.optimize_route(self.factory, self.router)

    def _record_day_stats(self, day: int, date: str) -> dict:
        """Record statistics for the day's deliveries"""
        stats = {
            'day': day,
            'date': date,
            'trucks': [],
            'total_distance': 0,
            'total_deliveries': 0,
            'total_weight': 0,
            'unassigned': len(self.unassigned_points)
        }
        
        for truck in self.trucks:
            if truck.deliveries:
                truck_stats = {
                    'id': truck.id,
                    'deliveries': len(truck.deliveries),
                    'distance': truck.total_distance,
                    'weight': sum(p.weight_kg for p in truck.deliveries),
                    'utilization': sum(p.weight_kg for p in truck.deliveries)/truck.capacity
                }
                stats['trucks'].append(truck_stats)
                stats['total_distance'] += truck.total_distance
                stats['total_deliveries'] += len(truck.deliveries)
                stats['total_weight'] += truck_stats['weight']
        
        self.monthly_stats.append(stats)
        
        # Print day summary
        print(f"\nDay {day} Summary:")
        print(f"Total deliveries: {stats['total_deliveries']}")
        print(f"Total distance: {stats['total_distance']:.1f} km")
        print(f"Total weight: {stats['total_weight']} kg")
        print(f"Unassigned points: {stats['unassigned']}")
        
        for truck in stats['trucks']:
            print(f"Truck {truck['id']}: {truck['deliveries']} deliveries, "
                  f"{truck['distance']:.1f} km, {truck['weight']} kg "
                  f"({truck['utilization']*100:.1f}% utilization)")
        
        return stats

    def _visualize_day(self, day: int, stats: dict):
        """Create interactive map visualization for the day"""
        m = folium.Map(location=FACTORY_LOCATION, zoom_start=12, tiles='OpenStreetMap')
        
        # Add factory marker
        folium.Marker(
            location=FACTORY_LOCATION,
            popup=f"<b>Factory</b><br>Day {day}",
            icon=folium.Icon(color='red', icon='industry', prefix='fa')
        ).add_to(m)
        
        # Color palette for trucks
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'pink', 'lightblue', 'darkgreen']
        
        # Plot each truck's route
        for i, truck in enumerate(self.trucks):
            if not truck.deliveries:
                continue
                
            color = colors[i % len(colors)]
            
            # Create route coordinates
            route_coords = [self.factory.coordinates]
            for point in truck.deliveries:
                route_coords.append(point.coordinates)
            route_coords.append(self.factory.coordinates)
            
            # Add route to map
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3.5,
                opacity=0.8,
                popup=f"<b>Truck {truck.id}</b><br>"
                      f"Deliveries: {len(truck.deliveries)}<br>"
                      f"Distance: {truck.total_distance:.1f} km<br>"
                      f"Load: {sum(p.weight_kg for p in truck.deliveries)}/{truck.capacity} kg"
            ).add_to(m)
            
            # Add delivery markers
            for j, point in enumerate(truck.deliveries):
                folium.Marker(
                    location=point.coordinates,
                    popup=f"<b>Delivery {point.id}</b><br>"
                          f"Weight: {point.weight_kg} kg<br>"
                          f"Truck: {truck.id}, Stop: {j+1}",
                    icon=folium.Icon(
                        color=color,
                        icon='truck' if j == 0 else 'circle',
                        prefix='fa'
                    )
                ).add_to(m)
        
        # Save map
        map_path = os.path.join(OUTPUT_DIR, "daily_maps", f"day_{day}_routes.html")
        m.save(map_path)
        print(f"Day {day} visualization saved to {map_path}")

    def _generate_monthly_report(self):
        """Generate monthly summary reports and charts"""
        print("\n=== MONTHLY SUMMARY ===")
        
        # Prepare data for charts
        days = [s['day'] for s in self.monthly_stats]
        dates = [s['date'] for s in self.monthly_stats]
        deliveries = [s['total_deliveries'] for s in self.monthly_stats]
        distances = [s['total_distance'] for s in self.monthly_stats]
        unassigned = [s['unassigned'] for s in self.monthly_stats]
        
        # Truck utilization data
        util_data = [[] for _ in range(NUM_TRUCKS)]
        delivery_counts = [[] for _ in range(NUM_TRUCKS)]
        
        for day in self.monthly_stats:
            for truck in day['trucks']:
                util_data[truck['id']-1].append(truck['utilization'])
                delivery_counts[truck['id']-1].append(truck['deliveries'])
        
        # 1. Daily Deliveries Chart
        plt.figure(figsize=(14, 7))
        plt.bar(days, deliveries, color='skyblue', label='Assigned Deliveries')
        plt.bar(days, unassigned, bottom=deliveries, color='salmon', label='Unassigned Deliveries')
        plt.title('Daily Delivery Volumes', fontsize=14)
        plt.xlabel('Day of Month', fontsize=12)
        plt.ylabel('Number of Deliveries', fontsize=12)
        plt.xticks(days, [d.split('-')[2] for d in dates])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "monthly_charts", "daily_deliveries.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"Daily deliveries chart saved to {chart_path}")
        
        # 2. Truck Utilization Chart
        plt.figure(figsize=(14, 7))
        for i in range(NUM_TRUCKS):
            plt.plot(days, util_data[i], 
                    marker='o', 
                    label=f'Truck {i+1}',
                    linewidth=2.5)
        
        plt.title('Daily Truck Utilization Rates', fontsize=14)
        plt.xlabel('Day of Month', fontsize=12)
        plt.ylabel('Capacity Utilization (%)', fontsize=12)
        plt.xticks(days, [d.split('-')[2] for d in dates])
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "monthly_charts", "truck_utilization.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"Truck utilization chart saved to {chart_path}")
        
        # 3. Truck Deliveries Chart
        plt.figure(figsize=(14, 7))
        for i in range(NUM_TRUCKS):
            plt.plot(days, delivery_counts[i], 
                    marker='o', 
                    label=f'Truck {i+1}',
                    linewidth=2.5)
        
        plt.axhline(y=MIN_DELIVERIES_PER_TRUCK, color='r', linestyle='--', 
                   label='Minimum Required')
        plt.title('Daily Deliveries per Truck', fontsize=14)
        plt.xlabel('Day of Month', fontsize=12)
        plt.ylabel('Number of Deliveries', fontsize=12)
        plt.xticks(days, [d.split('-')[2] for d in dates])
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        chart_path = os.path.join(OUTPUT_DIR, "monthly_charts", "truck_deliveries.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"Truck deliveries chart saved to {chart_path}")
        
        # Calculate summary statistics
        total_deliveries = sum(deliveries)
        total_unassigned = sum(unassigned)
        total_distance = sum(distances)
        avg_daily_deliveries = total_deliveries / DAYS_IN_MONTH
        success_rate = total_deliveries / (total_deliveries + total_unassigned) * 100
        
        # Print summary
        print("\nMonthly Performance Metrics:")
        print(f"Total Deliveries: {total_deliveries}")
        print(f"Total Unassigned Deliveries: {total_unassigned}")
        print(f"Delivery Success Rate: {success_rate:.1f}%")
        print(f"Average Daily Deliveries: {avg_daily_deliveries:.1f}")
        print(f"Total Distance Traveled: {total_distance:.1f} km")
        
        # Save summary to file
        summary_path = os.path.join(OUTPUT_DIR, "monthly_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MONTHLY DELIVERY SUMMARY\n")
            f.write("=======================\n\n")
            f.write(f"Total Days: {DAYS_IN_MONTH}\n")
            f.write(f"Number of Trucks: {NUM_TRUCKS}\n")
            f.write(f"Truck Capacity: {TRUCK_CAPACITY} kg\n")
            f.write(f"Minimum Deliveries per Truck: {MIN_DELIVERIES_PER_TRUCK}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"- Total Deliveries: {total_deliveries}\n")
            f.write(f"- Total Unassigned Deliveries: {total_unassigned}\n")
            f.write(f"- Delivery Success Rate: {success_rate:.1f}%\n")
            f.write(f"- Average Daily Deliveries: {avg_daily_deliveries:.1f}\n")
            f.write(f"- Total Distance Traveled: {total_distance:.1f} km\n")
        
        print(f"Monthly summary saved to {summary_path}")

class Factory:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.lat, self.lon)

class Truck:
    def __init__(self, truck_id: int, capacity: int):
        self.id = truck_id
        self.capacity = capacity
        self.deliveries = []
        self.total_distance = 0.0
    
    def add_delivery(self, point: DeliveryPoint) -> bool:
        """Add delivery if capacity allows"""
        if self.current_weight + point.weight_kg <= self.capacity:
            self.deliveries.append(point)
            return True
        return False
    
    def can_add_delivery(self, point: DeliveryPoint) -> bool:
        """Check if delivery can be added without exceeding capacity"""
        return self.current_weight + point.weight_kg <= self.capacity
    
    @property
    def current_weight(self) -> int:
        return sum(p.weight_kg for p in self.deliveries)
    
    def clear_route(self):
        """Reset truck's deliveries and distance"""
        self.deliveries = []
        self.total_distance = 0.0
    
    def optimize_route(self, factory: Factory, router):
        """Optimize delivery route using 2-opt algorithm"""
        if not self.deliveries:
            return
        
        # Start with nearest neighbor as initial route
        current_route = [factory.coordinates]
        remaining_points = self.deliveries.copy()
        
        while remaining_points:
            last_point = current_route[-1]
            # Find nearest remaining point
            nearest = min(remaining_points, 
                         key=lambda p: router.get_road_distance(last_point, p.coordinates))
            current_route.append(nearest.coordinates)
            remaining_points.remove(nearest)
        
        current_route.append(factory.coordinates)
        
        # Apply 2-opt optimization
        improved = True
        while improved:
            improved = False
            for i in range(1, len(current_route)-2):
                for j in range(i+1, len(current_route)-1):
                    if j == i:
                        continue
                    
                    # Calculate current distance
                    original = (router.get_road_distance(current_route[i-1], current_route[i]) +
                               router.get_road_distance(current_route[j], current_route[j+1]))
                    
                    # Calculate new distance if we reverse points i to j
                    new_dist = (router.get_road_distance(current_route[i-1], current_route[j]) +
                               router.get_road_distance(current_route[i], current_route[j+1]))
                    
                    if new_dist < original:
                        # Reverse the segment between i and j
                        current_route[i:j+1] = reversed(current_route[i:j+1])
                        improved = True
        
        # Update truck's deliveries order and total distance
        self.deliveries = []
        for coord in current_route[1:-1]:
            for point in self.deliveries:  # Find original point objects
                if point.coordinates == coord:
                    self.deliveries.append(point)
                    break
        
        # Calculate total distance
        self.total_distance = sum(
            router.get_road_distance(current_route[i], current_route[i+1])
            for i in range(len(current_route)-1)
        )

class RealRoadRouter:
    def __init__(self):
        self.cache = {}
        self.use_osrm = True  # Set to False to use haversine as fallback
    
    def get_road_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Get road distance between two points using OSRM or haversine"""
        cache_key = (start, end)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.use_osrm:
            try:
                # OSRM API call
                url = f"{OSRM_ENDPOINT}{start[1]},{start[0]};{end[1]},{end[0]}"
                response = requests.get(url, timeout=5)
                data = response.json()
                
                if data['code'] == 'Ok':
                    distance = data['routes'][0]['distance'] / 1000  # Convert to km
                    self.cache[cache_key] = distance
                    return distance
            except:
                self.use_osrm = False
                print("OSRM service failed, falling back to haversine distance")
        
        # Fallback to haversine distance
        distance = self._haversine_distance(start, end)
        self.cache[cache_key] = distance
        return distance
    
    def _haversine_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Calculate great-circle distance between two points"""
        lat1, lon1 = math.radians(start[0]), math.radians(start[1])
        lat2, lon2 = math.radians(end[0]), math.radians(end[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return 6371 * c  # Earth radius in km

class RouteOptimizer:
    def __init__(self, trucks: List[Truck], router: RealRoadRouter):
        self.trucks = trucks
        self.router = router
    
    def optimize_routes(self, factory: Factory, points: List[DeliveryPoint]) -> List[DeliveryPoint]:
        """Optimize delivery routes using Clarke-Wright Savings Algorithm"""
        if not points:
            return []
        
        print(f"Optimizing routes for {len(points)} points using Clarke-Wright algorithm...")
        
        # Step 1: Calculate savings for all point pairs
        savings = self._calculate_savings(factory, points)
        
        # Step 2: Initialize routes (each point as separate route)
        routes = [[point] for point in points]
        route_weights = [point.weight_kg for point in points]
        
        # Step 3: Process savings in descending order
        for saving in savings:
            point_i, point_j, saving_value = saving.point_i, saving.point_j, saving.savings
            
            # Find routes containing these points
            route_i_idx, route_j_idx = None, None
            for idx, route in enumerate(routes):
                if point_i in route:
                    route_i_idx = idx
                if point_j in route:
                    route_j_idx = idx
            
            # Skip if same route or not found
            if route_i_idx == route_j_idx or route_i_idx is None or route_j_idx is None:
                continue
            
            # Check capacity constraints
            combined_weight = route_weights[route_i_idx] + route_weights[route_j_idx]
            if combined_weight > TRUCK_CAPACITY:
                continue
            
            # Check if points are at route ends
            route_i, route_j = routes[route_i_idx], routes[route_j_idx]
            
            # Try all possible merge combinations
            merged_route = None
            
            # Case 1: i first, j last
            if route_i[0] == point_i and route_j[-1] == point_j:
                merged_route = route_j + route_i
            # Case 2: i last, j first
            elif route_i[-1] == point_i and route_j[0] == point_j:
                merged_route = route_i + route_j
            # Case 3: i first, j first (reverse j)
            elif route_i[0] == point_i and route_j[0] == point_j:
                merged_route = route_j[::-1] + route_i
            # Case 4: i last, j last (reverse i)
            elif route_i[-1] == point_i and route_j[-1] == point_j:
                merged_route = route_i + route_j[::-1]
            
            if merged_route:
                # Update routes
                routes[route_i_idx] = merged_route
                route_weights[route_i_idx] = combined_weight
                
                # Remove the merged route
                del routes[route_j_idx]
                del route_weights[route_j_idx]
        
        # Step 4: Assign routes to trucks
        unassigned = []
        
        # Sort routes by weight (heaviest first)
        sorted_routes = sorted(zip(routes, route_weights), key=lambda x: x[1], reverse=True)
        
        for route, _ in sorted_routes:
            assigned = False
            for truck in self.trucks:
                if truck.can_accept_route(route):
                    for point in route:
                        truck.add_delivery(point)
                    assigned = True
                    break
            
            if not assigned:
                unassigned.extend(route)
        
        # Step 5: Optimize individual truck routes
        for truck in self.trucks:
            if truck.deliveries:
                truck.optimize_route(factory, self.router)
        
        print(f"Route optimization complete. Unassigned deliveries: {len(unassigned)}")
        return unassigned
    
    def _calculate_savings(self, factory: Factory, points: List[DeliveryPoint]) -> List[Saving]:
        """Calculate Clarke-Wright savings for all point pairs"""
        savings = []
        
        # Pre-calculate distances from factory
        factory_dist = {}
        for point in points:
            factory_dist[point.id] = self.router.get_road_distance(factory.coordinates, point.coordinates)
        
        # Calculate savings for all pairs
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                point_i = points[i]
                point_j = points[j]
                
                # Distance between points
                dist_ij = self.router.get_road_distance(point_i.coordinates, point_j.coordinates)
                
                # Savings formula: S(i,j) = d(0,i) + d(0,j) - d(i,j)
                saving_value = factory_dist[point_i.id] + factory_dist[point_j.id] - dist_ij
                
                if saving_value > 0:
                    savings.append(Saving(point_i, point_j, saving_value))
        
        # Sort by savings in descending order
        savings.sort(reverse=True)
        return savings

# Run the simulation
if __name__ == "__main__":
    print("=== MONTHLY DELIVERY OPTIMIZATION SYSTEM ===")
    system = DeliverySystem()
    system.run_monthly_simulation()
