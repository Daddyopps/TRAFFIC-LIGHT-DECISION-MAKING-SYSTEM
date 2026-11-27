import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class TrafficFuzzyController:
    def __init__(self):
        # 1. DEFINE THE FUZZY VARIABLES (Inputs and Outputs)
        # Inputs: Queue length on the Main Road, and Queue length on the Cross Road
        self.main_queue = ctrl.Antecedent(np.arange(0, 51, 1), 'main_queue')
        self.cross_queue = ctrl.Antecedent(np.arange(0, 51, 1), 'cross_queue')
        
        # Output: How many seconds the Green Light should stay on
        self.green_time = ctrl.Consequent(np.arange(5, 61, 1), 'green_time')

        # 2. CREATE MEMBERSHIP FUNCTIONS (Low, Medium, High)
        # Main Road
        self.main_queue['low'] = fuzz.trimf(self.main_queue.universe, [0, 0, 15])
        self.main_queue['medium'] = fuzz.trimf(self.main_queue.universe, [10, 25, 40])
        self.main_queue['high'] = fuzz.trimf(self.main_queue.universe, [30, 50, 50])

        # Cross Road
        self.cross_queue['low'] = fuzz.trimf(self.cross_queue.universe, [0, 0, 15])
        self.cross_queue['medium'] = fuzz.trimf(self.cross_queue.universe, [10, 25, 40])
        self.cross_queue['high'] = fuzz.trimf(self.cross_queue.universe, [30, 50, 50])

        # Green Light Times
        self.green_time['very_short'] = fuzz.trimf(self.green_time.universe, [5, 5, 15])
        self.green_time['short'] = fuzz.trimf(self.green_time.universe, [10, 20, 30])
        self.green_time['average'] = fuzz.trimf(self.green_time.universe, [25, 40, 50])
        self.green_time['long'] = fuzz.trimf(self.green_time.universe, [45, 60, 60])

        # 3. THE FUZZY RULES (The AI Brain Logic)
        # Main road is LOW (0 - 15 cars)
        rule1 = ctrl.Rule(self.main_queue['low'] & self.cross_queue['low'], self.green_time['very_short'])
        rule2 = ctrl.Rule(self.main_queue['low'] & self.cross_queue['medium'], self.green_time['very_short'])
        rule3 = ctrl.Rule(self.main_queue['low'] & self.cross_queue['high'], self.green_time['very_short'])

        # Main road is MEDIUM (10 - 40 cars)
        rule4 = ctrl.Rule(self.main_queue['medium'] & self.cross_queue['low'], self.green_time['average'])
        rule5 = ctrl.Rule(self.main_queue['medium'] & self.cross_queue['medium'], self.green_time['short']) 
        rule6 = ctrl.Rule(self.main_queue['medium'] & self.cross_queue['high'], self.green_time['very_short']) 

        # Main road is HIGH (30 - 50 cars)
        rule7 = ctrl.Rule(self.main_queue['high'] & self.cross_queue['low'], self.green_time['long'])
        rule8 = ctrl.Rule(self.main_queue['high'] & self.cross_queue['medium'], self.green_time['average'])
        rule9 = ctrl.Rule(self.main_queue['high'] & self.cross_queue['high'], self.green_time['average']) 

        # 4. BUILD THE CONTROL SYSTEM
        self.traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.traffic_sim = ctrl.ControlSystemSimulation(self.traffic_ctrl)

    def calculate_green_time(self, main_cars: int, cross_cars: int) -> int:
        # Safety limit to prevent math crashing if count exceeds 50
        main_cars = min(max(main_cars, 0), 50)
        cross_cars = min(max(cross_cars, 0), 50)

        # Pass the inputs into the Fuzzy System
        self.traffic_sim.input['main_queue'] = main_cars
        self.traffic_sim.input['cross_queue'] = cross_cars

        # Run the math (Defuzzification)
        self.traffic_sim.compute()

        # Extract the exact seconds and round to a whole number
        calculated_time = self.traffic_sim.output['green_time']
        return int(round(calculated_time))

# Quick Test 
if __name__ == "__main__":
    ai = TrafficFuzzyController()
    print("Testing Updated Fuzzy Logic Brain...")
    print(f"5 main, 2 cross -> Green Time: {ai.calculate_green_time(5, 2)} seconds")
    print(f"12 main, 35 cross -> Green Time: {ai.calculate_green_time(12, 35)} seconds")
    print(f"45 main, 2 cross -> Green Time: {ai.calculate_green_time(45, 2)} seconds")
    print(f"45 main, 45 cross -> Green Time: {ai.calculate_green_time(45, 45)} seconds")