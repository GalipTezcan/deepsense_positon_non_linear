from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


def read_lat_lon_from_file(array, data_root: Path | str, scenario_index: int) -> np.ndarray:
	lat, lon = np.zeros(len(array)).astype(np.float64), np.zeros(len(array)).astype(np.float64)
	root = Path(data_root)
	try:
		for i, path in enumerate(array):
			with open(root / f"Scenario{scenario_index}"  / path.split("./")[-1], "r") as f:
				content = f.read().strip()
				lat[i], lon[i] = map(np.float64, content.split("\n"))
		return np.column_stack([lat, lon])
	except Exception as exc:
		print(f"Error reading lat/lon from file list: {exc}")
		return np.column_stack([lat, lon])


def read_pwr_from_file(array, data_root: Path | str, scenario_index: int) -> np.ndarray:
	pwr: list[list[str]] = []
	root = Path(data_root)
	for path in array:
		try:
			with open(root / f"Scenario{scenario_index}"  / path.split("./")[-1], "r") as f:
				content = f.read().strip()
				pwr.append(content.split("\n"))
		except Exception as exc:
			print(f"Error reading pwr from file '{path}': {exc}")
			pwr.append([np.nan for _ in range(64)])
	return np.array(pwr).astype(np.float64)


def haversine(lat1, lon1, lat2, lon2) -> Tuple[np.ndarray, np.ndarray]:
	# All args in degrees
	R = 6371000  # Earth radius in meters
	lat1 = np.radians(lat1)
	lon1 = np.radians(lon1)
	lat2 = np.radians(lat2)
	lon2 = np.radians(lon2)
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
	return R * c, c

def get_bearing(lat1, lon1, lat2, lon2):
    dLon = np.radians(lon2 - lon1)
    y = np.sin(dLon) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dLon)
    return np.degrees(np.arctan2(y, x))

def load_scenario(scenario_index: int, data_root: Path | str = "Position", return_pwr: bool = True,N_PWR:int=64) -> pd.DataFrame:
	"""Load and process the scenario CSV for a deepsense scenario.

	This function expects the following file inside
	data_root/Scenario{scenario_index}:
	- scenario{index}.csv

	Return scenario CSV with derived features.
	"""
	print("********************************************************")
	print(f"Loading Scenario{scenario_index} dataset from {data_root}")
	root = Path(data_root)
	scenario_dir = root / f"scenario{scenario_index}"
	if not scenario_dir.exists():
		raise FileNotFoundError(f"development_dataset directory not found for Scenario{scenario_index}: {scenario_dir}")

	prefix = f"scenario{scenario_index}_dev"
	scenario_csv = scenario_dir / f"{prefix}.csv"
	if not scenario_csv.exists():
		raise FileNotFoundError(f"Missing expected CSV: {scenario_csv}")
	# Load scenario CSV
	scenario_df = pd.read_csv(scenario_csv)

	# Convert time_stamp to datetime
	scenario_df["time_stamp"] = pd.to_datetime(scenario_df["time_stamp"], format='%H:%M:%S-%f')
	# Fill NaN values by interpolating based on before and after rows (linear interpolation)
	# Axis=0 for row-wise interpolation; limit_direction='both' fills NaNs at start/end if possible
	scenario_df.interpolate(method='linear', axis=0, limit_direction='both',inplace=True)
	# Convert unit2_DGPS to 0/1 if it is not already
	if "unit2_DGPS" in scenario_df.columns:
		scenario_df["unit2_DGPS"] = scenario_df["unit2_DGPS"].map({"Yes": 1, "No": 0,1:1,0:0})
		scenario_df["unit2_DGPS"]=scenario_df["unit2_DGPS"].astype(int)


	# Extract locations for unit1 and unit2
	scenario_df[["unit1_lat", "unit1_lon"]] = read_lat_lon_from_file(scenario_df["unit1_loc"], root, scenario_index)
	scenario_df[["unit2_lat", "unit2_lon"]] = read_lat_lon_from_file(scenario_df["unit2_loc"], root, scenario_index)

	scenario_df["bearing"] = get_bearing(scenario_df["unit1_lat"], scenario_df["unit1_lon"], scenario_df["unit2_lat"], scenario_df["unit2_lon"])

	scenario_df["unit1_unit2_distance"] = haversine(
		scenario_df["unit1_lat"],
		scenario_df["unit1_lon"],
		scenario_df["unit2_lat"],
		scenario_df["unit2_lon"],
	)[0]

	scenario_df["unit2_lat_prev"] = scenario_df["unit2_lat"].shift(1).fillna(scenario_df["unit2_lat"].iloc[0])
	scenario_df["unit2_lon_prev"] = scenario_df["unit2_lon"].shift(1).fillna(scenario_df["unit2_lon"].iloc[0])
	scenario_df["unit2_prev_distance"] = haversine(
		scenario_df["unit2_lat_prev"],
		scenario_df["unit2_lon_prev"],
		scenario_df["unit2_lat"],
		scenario_df["unit2_lon"],
	)[0]


	droplist=["index","unit1_rgb","unit1_pwr_60ghz","unit1_loc","unit2_loc","time_stamp","unit1_beam","unit1_max_pwr","unit2_lat_prev","unit2_lon_prev"]
	# Optionally read and append power features (pwr_0..pwr_63) from unit1_pwr_60ghz
	pwr_arr = read_pwr_from_file(scenario_df["unit1_pwr_60ghz"], root, scenario_index)[:,[i*64//N_PWR for i in range(N_PWR)]]

	if return_pwr:
		pwr_df = pd.DataFrame(pwr_arr, columns=[f"pwr_{i}" for i in range(N_PWR)])
		scenario_df = pd.concat([scenario_df, pwr_df], axis=1)
		if "unit1_beam_index" in scenario_df.columns:
			droplist.append("unit1_beam_index")	
	else:
			scenario_df["unit1_beam_index"]=pwr_arr.argmax(axis=1)
	
	scenario_df["unit1_beam_index_f"]=scenario_df["unit1_beam_index"].copy().astype(float)
	scenario_df.dropna(axis=1, how='all',inplace=True)
	for col in ["unit2_sat_used","unit2_fix_type","unit1_lidar","unit1_lidar_SCR","unit1_radar","unit2_loc_cal"]:
		if col in scenario_df.columns:
			droplist.append(col)
	scenario_df.drop(columns=droplist,inplace=True)
	# Drop constant columns (columns with only one unique value)
	nunique = scenario_df.nunique(dropna=False)
	constant_cols = nunique[nunique <= 1].index.tolist()
	if constant_cols:
		scenario_df.drop(columns=constant_cols, inplace=True)
	return scenario_df


__all__ = ["load_scenario"]


