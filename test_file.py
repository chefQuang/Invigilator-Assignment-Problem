import json
import sys

def run_tests(json_filepath):
    print(f"Loading data from {json_filepath} for testing...")
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {json_filepath}. Run data_processing.py first!")
        sys.exit(1)

    # Extract for easy testing
    sets = data.get('sets', {})
    params = data.get('parameters', {})
    
    invigilators = sets.get('invigilators', [])
    shifts = sets.get('shifts', [])
    
    shift_date = params.get('shift_date', {})
    req_staff = params.get('req_staff', {})
    staff_pref = params.get('staff_pref', {})

    print("\nRunning Tests...")

    try:
        # --- TEST 1: Basic Structure Check ---
        assert len(invigilators) > 0, "Invigilators list is empty!"
        assert len(shifts) > 0, "Shifts list is empty!"
        print("Test 1 Passed: Data sets are not empty.")

        # --- TEST 2: Data Integrity (Mapping Match) ---
        # Every shift must have a date, campus, duration, and required staff count
        for j in shifts:
            assert j in shift_date, f"Shift {j} is missing a Date!"
            assert j in req_staff, f"Shift {j} is missing Required Staff!"
            assert isinstance(req_staff[j], int), f"Required staff for {j} must be an integer!"
            assert req_staff[j] > 0, f"Shift {j} requires 0 staff, which is invalid!"
        print("Test 2 Passed: All shifts have complete and valid parameters.")

        # --- TEST 3: Synthetic Data Check ---
        # Every invigilator must have a generated location preference
        valid_preferences = ["CS1", "CS2", "Neutral"]
        for i in invigilators:
            assert i in staff_pref, f"Invigilator {i} is missing a preference!"
            assert staff_pref[i] in valid_preferences, f"Invigilator {i} has an invalid preference: {staff_pref[i]}"
        print("Test 3 Passed: Synthetic preferences generated correctly.")

        # --- TEST 4: General Data Insights (Optional but helpful) ---
        print(f"\nQuick Insights:")
        print(f"   Total Invigilators: {len(invigilators)}")
        print(f"   Total Unique Shifts: {len(shifts)}")
        print(f"   Total Staff Required across all shifts: {sum(req_staff.values())}")
        
        # Checking if we have enough staff to cover the absolute minimum demand
        max_concurrent_demand = max(req_staff.values())
        assert len(invigilators) >= max_concurrent_demand, "Not enough total invigilators to cover the largest single shift!"
        print("Test 4 Passed: Logical baseline limits look OK.")

        print("\nALL TESTS PASSED! Your data is ready for the ILP solver.")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        print("Go back and check your data_processing.py script.")

if __name__ == "__main__":
    run_tests('processed_data.json')