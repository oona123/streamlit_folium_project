import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import folium

accel_data = pd.read_csv("https://raw.githubusercontent.com/oona123/streamlit_folium_project/main/Linear%20Accelerometer.csv?raw=true")
gps_data = pd.read_csv("https://raw.githubusercontent.com/oona123/streamlit_folium_project/main/Location.csv?raw=true")

st.title("Liikunnan analyysi ja visualisointi")
st.write("Tässä projektissa analysoimme kävelyliikettä kiihtyvyys- ja GPS-datan avulla.")

#laskettu ipynb- tiedostossa
step_count_filtered = 382
step_count_fourier = 396
average_speed = 1.62
distance_travelled = 332.08
stride_length = 0.87

#tulostetaan tiedot
st.write("#### Lasketut tiedot:")
st.write(f"- **Askelmäärä suodatetusta kiihtyvyysdatasta**: {step_count_filtered} askelta")
st.write(f"- **Askelmäärä Fourier-analyysin perusteella**: {step_count_fourier} askelta")
st.write(f"- **Keskinopeus**: {average_speed} m/s")
st.write(f"- **Kuljettu matka**: {distance_travelled} m")
st.write(f"- **Askelpituus**: {stride_length} m")

def butter_lowpass_filter(data, cutoff=2.5, fs=50, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

filtered_acc = butter_lowpass_filter(accel_data["Z (m/s^2)"])

st.write("#### Suodatettu kiihtyvyysdata:")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(accel_data["Time (s)"], filtered_acc, label="Suodatettu Z-kiihtyvyys", linewidth=1.5)
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Kiihtyvyys (m/s²)")
ax.set_xlim(10,60)
ax.legend()
ax.grid(axis='y')
st.pyplot(fig)

#Fourier-analyysi
N = len(filtered_acc)
T = accel_data["Time (s)"].iloc[1] - accel_data["Time (s)"].iloc[0]
freqs = fftfreq(N, T)
fft_values = fft(filtered_acc)

st.write("#### Tehospektri Fourier-analyysillä:")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freqs[:N // 2], abs(fft_values[:N // 2]), label="Tehospektri")
ax.set_xlabel("Taajuus (Hz)")
ax.set_ylabel("Teho")
ax.grid(axis='y')
ax.legend()
st.pyplot(fig)

route = list(zip(gps_data["Latitude (°)"], gps_data["Longitude (°)"]))
m = folium.Map(location=[gps_data["Latitude (°)"].mean(), gps_data["Longitude (°)"].mean()], zoom_start=16)
folium.PolyLine(route, color="blue", weight=3,opacity=1).add_to(m)
st.write("#### Reitti kartalla:")
st.components.v1.html(m._repr_html_(), height=500)
