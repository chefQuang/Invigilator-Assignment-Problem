import pandas as pd
import json
import random

def process_and_export_data(csv_filepath, output_json_path):
    df = pd.read_csv(csv_filepath)

    role_amount = df.groupby(['MS Ca thi', 'Nhiệm vụ']).size().reset_index(name='Số luọng')
    print(role_amount.head(50))

    #1. EXTRACT SETS
    invigilators = df['MS của CÁN BỘ COI THI'].dropna().unique().tolist()
    shifts = df['MS Ca thi'].dropna().unique().tolist()
    dates = df['Ngày'].dropna().unique().tolist()
    df['Cơ sở'] = df['Cơ sở'].fillna('Chưa xác định')
    campuses = df['Cơ sở'].unique().tolist()

    #2. EXTRACT PARAMS
    shift_info_df = df.drop_duplicates(subset=['MS Ca thi'])
    shift_date = shift_info_df.set_index('MS Ca thi')['Ngày'].to_dict()
    shift_campus = shift_info_df.set_index('MS Ca thi')['Cơ sở'].to_dict()
    shift_duration = shift_info_df.set_index('MS Ca thi')['Thời gian'].to_dict()
    shift_time = shift_info_df.set_index('MS Ca thi')['GIỜ'].to_dict()

    req_staff = df.groupby('MS Ca thi')['MS của CÁN BỘ COI THI'].count().to_dict()


    #Randomly assign a location preference to each staff member
    preferences = ["Cơ sở 1", "Cơ sở 2", "Neutral"]
    staff_pref = {}
    for staff in invigilators:
        staff_pref[staff] = random.choice(preferences)

    processed_data = {
        "sets": {
            "invigilators": invigilators,
            "shifts": shifts,
            "dates": dates,
            "campuses": campuses,
        },
        "parameters": {
            "shift_date": shift_date,
            "shift_campus": shift_campus,
            "shift_duration": shift_duration,
            "shift_time": shift_time,
            "req_staff": req_staff,
            "staff_pref": staff_pref
        }
    }

    with open(output_json_path, 'w', encoding = 'utf-8') as f:
        json.dump(processed_data, f, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    INPUT_CSV = 'IAPDataset.csv'
    OUTPUT_JSON = 'processed_data.json'

    process_and_export_data(INPUT_CSV, OUTPUT_JSON)
