# import pandas as pd

# # Path to the CSV file
# csv_file_path = r"ProcessedData\AB01\normal_walk_1_0-6\AB01_normal_walk_1_0-6_emg.csv"

# # Read the CSV file into a Pandas DataFrame
# dataset = pd.read_csv(csv_file_path, index_col='time')

# # Access data using column names as keys
# # For example, to access the values in the 'LTA' column:
# values_LTA = dataset['LTA']

# # To access the values in the 'RTA' column:
# values_RTA = dataset['RTA']

# # Print the first few values for demonstration
# print("Values in LTA column:")
# print(values_LTA.head())

# print("\nValues in RTA column:")
# print(values_RTA.head())
import os
import pandas as pd
import pickle

# Define the directory containing the data folders
data_directory = "ProcessedData"

# Initialize the primary data structure
primary_data_structure = {
    'file names': [],
    'data': {},
    'metadata': {}
}

test_data_structure = {
    'file names': [],
    'data': {},
    'metadata': {}
}

# leave a subject out 

# leave a activity out


# Iterate through each folder in the data directory
for folder_name2 in os.listdir(data_directory): # AB0, AB1
    folder_path1 = os.path.join(data_directory, folder_name2)
    for folder_name in os.listdir(folder_path1): # normal_walk folder
        folder_path = os.path.join(folder_path1, folder_name)
        dfs = []
        nanfound = False
        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path) and folder_path.__contains__('AB05')==False:#(folder_path.__contains__('normal_walk_') or folder_path.__contains__('incline_walk_')  or folder_path.__contains__('jump_')) and (folder_path.__contains__('_skip')==False) and  (folder_path.__contains__('AB01')==True or folder_path.__contains__('AB02')==True or folder_path.__contains__('AB03')==True or folder_path.__contains__('AB04')==True ) :
            # Iterate through each file in the folder
            ###############################################
            files_ = sorted(os.listdir(folder_path))[-3]
            import numpy as np
            column = ["ankle_angle_l_moment"]
            df = pd.read_csv(os.path.join(folder_path, files_))
            select = df[column]
         
            if np.isnan(np.array(select)).any():
                continue
            ######################## NAN checks ###########    
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Check if the item in the directory is a file and ends with '.csv'
                if os.path.isfile(file_path) :#and (file_name.__contains__('_normal_walk_') or file_name.__contains__('_incline_walk_') or file_name.__contains__('_jump_')) and (file_name.__contains__('_skip')==False)  and (file_name.__contains__('_moment_filt') or file_name.__contains__('_angle') or file_name.__contains__('_emg') or file_name.__contains__('_imu_real') or file_name.__contains__('_velocity') ):
                    # Read the CSV file into a Pandas DataFrame
                    # print(file_path)
                 
                                
                        
                    if file_name.__contains__('_angle'):
                        column = ["time","hip_rotation_r","knee_angle_r","ankle_angle_r","hip_rotation_l"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)
                    elif file_name.__contains__('_moment_filt'):
                        column = ["knee_angle_l_moment","ankle_angle_l_moment"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)  
                        print("File read",file_path)  
                    elif file_name.__contains__('_emg4'):
                        # print("EMG READ############")
                        column = ["RTA","LRF","RRF","LBF","RBF","LGMED","RGMED","RMGAS","LVL","RVL","LGRAC","RGRAC","LGMAX","RGMAX"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)
                    elif file_name.__contains__('_imu_real'):
                        column = ["RShank_ACCX","RShank_ACCY","RShank_ACCZ","RShank_GYROX","RShank_GYROY","RShank_GYROZ","LAThigh_ACCX","LAThigh_ACCY","LAThigh_ACCZ","LAThigh_GYROX","LAThigh_GYROY","LAThigh_GYROZ","RAThigh_ACCX","RAThigh_ACCY","RAThigh_ACCZ","RAThigh_GYROX","RAThigh_GYROY","RAThigh_GYROZ","LPThigh_ACCX","LPThigh_ACCY","LPThigh_ACCZ","LPThigh_GYROX","LPThigh_GYROY","LPThigh_GYROZ","RPThigh_ACCX","RPThigh_ACCY","RPThigh_ACCZ","RPThigh_GYROX","RPThigh_GYROY","RPThigh_GYROZ","LPelvis_ACCX","LPelvis_ACCY","LPelvis_ACCZ","LPelvis_GYROX","LPelvis_GYROY","LPelvis_GYROZ","RPelvis_ACCX","RPelvis_ACCY","RPelvis_ACCZ","RPelvis_GYROX","RPelvis_GYROY","RPelvis_GYROZ"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)
                    elif file_name.__contains__('_velocity'):
                        column = ["hip_rotation_velocity_r","knee_velocity_r","ankle_velocity_r","hip_rotation_velocity_l"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)

                # df = pd.read_csv(file_path)
                
                
            # Set the column labels (if not already set)
            # Check if dfs list is not empty before concatenating
            if dfs:
                # Concatenate all DataFrames in the list along axis 1 (columns) to assemble the final DataFrame
                dfs = pd.concat(dfs, axis=1)
                # if 'time' not in dfs.columns:
                #     dfs = dfs.rename(columns={'Unnamed: 0': 'time'})
            else:
                print("No valid CSV files found.")
            # dfs = pd.concat(dfs, axis=1)
            # if 'time' not in dfs.columns:
            #     dfs = dfs.rename(columns={'Unnamed: 0': 'time'})
            
            # Append the file name to the list of file names
            primary_data_structure['file names'].append(folder_name2+folder_name)
            
            # Store the DataFrame in the 'data' dictionary with the file name as the key
            primary_data_structure['data'][folder_name2+folder_name] = dfs
            
            
            # Store metadata if available
            primary_data_structure['metadata'][folder_name2+folder_name] = {
                'folder_name': folder_name2,
                # 'file_path': file_path,
                # Add any additional metadata you want to include
            }

# Print the primary data structure
# print("Primary Data Structure:")
# print(primary_data_structure["data"]['AB01normal_walk_1_0-6'].keys())
# primary_data_structure = pd.DataFrame(primary_data_structure)
# # Export the DataFrame to an Excel file
# primary_data_structure.to_excel('loaded_dataframe.xlsx', index=False)
# primary_data_structure.to_pickle('data_structure.pkl')

# Define the filename for the pickle file
pickle_file = 'Data_train\datatrain2_structure.pkl'

# Open the file in binary write mode and save the dictionary using pickle.dump()
with open(pickle_file, 'wb') as f:
    pickle.dump(primary_data_structure, f)
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# df = data["data"]['AB10normal_walk_1_2-5']   
# df.to_excel('output.xlsx', index=False)


# Iterate through each folder in the data directory
for folder_name2 in os.listdir(data_directory): # AB0, AB1
    folder_path1 = os.path.join(data_directory, folder_name2)
    for folder_name in os.listdir(folder_path1): # normal_walk folder
        folder_path = os.path.join(folder_path1, folder_name)
        dfs = []
        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path) and folder_path.__contains__('AB05')==True and (folder_path.__contains__('incline_walk_2_up10')  or folder_path.__contains__('jump_1_2_vertical') or folder_path.__contains__('normal_walk_1_1-2') or folder_path.__contains__('stairs_1_1_up')) :#(folder_path.__contains__('normal_walk_1_1-2') or folder_path.__contains__('incline_walk_2_up10')  or folder_path.__contains__('jump_1_vertical')) and (folder_path.__contains__('_skip')==False) and folder_path.__contains__('AB01')==True :
            # Iterate through each file in the folder
            ###############################################
            files_ = sorted(os.listdir(folder_path))[-3]
            import numpy as np
            column = ["ankle_angle_l_moment"]
            df = pd.read_csv(os.path.join(folder_path, files_))
            select = df[column]
            if np.isnan(np.array(select)).any():
                continue
            ######################## NAN checks ###########   


            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Check if the item in the directory is a file and ends with '.csv'
                if os.path.isfile(file_path) :#and (file_name.__contains__('_normal_walk_') or file_name.__contains__('_incline_walk_')  or file_name.__contains__('_jump_')) and (file_name.__contains__('_skip')==False)  and (file_name.__contains__('_moment_filt') or file_name.__contains__('_angle') or file_name.__contains__('_emg') or file_name.__contains__('_imu_real') or file_name.__contains__('_velocity') ):
                    # Read the CSV file into a Pandas DataFrame
                    
                    # print(file_path)
                    
                    if file_name.__contains__('_angle'):
                        column = ["time","hip_rotation_r","knee_angle_r","ankle_angle_r","hip_rotation_l"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)
                    elif file_name.__contains__('_moment_filt'):
                        column = ["knee_angle_l_moment","ankle_angle_l_moment"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)  
                        print("File read",file_path)  
                    elif file_name.__contains__('_emg4'):
                        column = ["RTA","LRF","RRF","LBF","RBF","LGMED","RGMED","RMGAS","LVL","RVL","LGRAC","RGRAC","LGMAX","RGMAX"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)
                    elif file_name.__contains__('_imu_real'):
                        column = ["RShank_ACCX","RShank_ACCY","RShank_ACCZ","RShank_GYROX","RShank_GYROY","RShank_GYROZ","LAThigh_ACCX","LAThigh_ACCY","LAThigh_ACCZ","LAThigh_GYROX","LAThigh_GYROY","LAThigh_GYROZ","RAThigh_ACCX","RAThigh_ACCY","RAThigh_ACCZ","RAThigh_GYROX","RAThigh_GYROY","RAThigh_GYROZ","LPThigh_ACCX","LPThigh_ACCY","LPThigh_ACCZ","LPThigh_GYROX","LPThigh_GYROY","LPThigh_GYROZ","RPThigh_ACCX","RPThigh_ACCY","RPThigh_ACCZ","RPThigh_GYROX","RPThigh_GYROY","RPThigh_GYROZ","LPelvis_ACCX","LPelvis_ACCY","LPelvis_ACCZ","LPelvis_GYROX","LPelvis_GYROY","LPelvis_GYROZ","RPelvis_ACCX","RPelvis_ACCY","RPelvis_ACCZ","RPelvis_GYROX","RPelvis_GYROY","RPelvis_GYROZ"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)
                    elif file_name.__contains__('_velocity'):
                        column = ["hip_rotation_velocity_r","knee_velocity_r","ankle_velocity_r","hip_rotation_velocity_l"]
                        df = pd.read_csv(file_path)
                        select = df[column]
                        dfs.append(select)
                        print("File read",file_path)

                # df = pd.read_csv(file_path)
                
                
            # Set the column labels (if not already set)
            # Check if dfs list is not empty before concatenating
            if dfs:
                # Concatenate all DataFrames in the list along axis 1 (columns) to assemble the final DataFrame
                dfs = pd.concat(dfs, axis=1)
                # if 'time' not in dfs.columns:
                #     dfs = dfs.rename(columns={'Unnamed: 0': 'time'})
            else:
                print("No valid CSV files found.")
            # dfs = pd.concat(dfs, axis=1)
            # if 'time' not in dfs.columns:
            #     dfs = dfs.rename(columns={'Unnamed: 0': 'time'})
            
            # Append the file name to the list of file names
            test_data_structure['file names'].append(folder_name2+folder_name)
            
            # Store the DataFrame in the 'data' dictionary with the file name as the key
            test_data_structure['data'][folder_name2+folder_name] = dfs
            
            
            # Store metadata if available
            test_data_structure['metadata'][folder_name2+folder_name] = {
                'folder_name': folder_name2,
                # 'file_path': file_path,
                # Add any additional metadata you want to include
            }

# Define the filename for the pickle file
pickle_file = 'Data_train/test_structure.pkl'

# Open the file in binary write mode and save the dictionary using pickle.dump()
with open(pickle_file, 'wb') as f:
    pickle.dump(test_data_structure, f)
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# df = data["data"]['AB10normal_walk_1_2-5']   
# df.to_excel('output.xlsx', index=False)

# "RTA","RRF","RBF","RGMED","RMGAS","RVL","RGRAC","RGMAX"
#"RShank_ACCX","RShank_ACCY","RShank_ACCZ","RShank_GYROX","RShank_GYROY","RShank_GYROZ","RPelvis_ACCX","RPelvis_ACCY","RPelvis_ACCZ","RPelvis_GYROX","RPelvis_GYROY","RPelvis_GYROZ"