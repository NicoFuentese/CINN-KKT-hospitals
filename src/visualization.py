import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os

#ploteo de carta gantt
def plot_advanced_gantt(df, makespan, J, setup_mins=10.0, output_path="data/processed/gantt_final.png"):
    fig, ax = plt.subplots(figsize=(20, 12))
    colors = plt.cm.get_cmap('tab20', J)
    
    for _, row in df.iterrows():
        start = row['real_start']
        dur_med = row['dur_medical']
        dur_occ = row['dur_occupancy']
        y = row['global_machine_id']
        j = int(row['job_id'])
        
        buffer_total = dur_occ - dur_med
        setup = setup_mins if buffer_total >= setup_mins else buffer_total / 2.0
        cleanup = buffer_total - setup
        
        # Setup
        ax.add_patch(patches.Rectangle((start, y - 0.4), setup, 0.8, 
                                       linewidth=0.5, edgecolor='black', facecolor='#FFD700', alpha=0.8))
        # Cirugía
        ax.add_patch(patches.Rectangle((start + setup, y - 0.4), dur_med, 0.8, 
                                     linewidth=1, edgecolor='black', facecolor=colors(j), alpha=0.9))
        # Cleanup
        ax.add_patch(patches.Rectangle((start + setup + dur_med, y - 0.4), cleanup, 0.8, 
                                       linewidth=0.5, edgecolor='black', facecolor='#FF6347', alpha=0.8))
        
        if dur_med > 30:
            ax.text(start + setup + dur_med/2, y, f"P{j}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # Leyenda
    setup_patch = patches.Patch(color='#FFD700', alpha=0.8, label='Setup (Preparación Sala)')
    med_patch = patches.Patch(color='gray', alpha=0.9, label='Cirugía / Intervención')
    clean_patch = patches.Patch(color='#FF6347', alpha=0.8, label='Cleanup (Limpieza)')
    ax.legend(handles=[setup_patch, med_patch, clean_patch], loc='upper right', fontsize=12)

    ax.set_yticks(range(1, 13))
    labels = []
    for r in range(1, 13):
        if r <= 4: labels.append(f"PRE-{r}")
        elif r <= 8: labels.append(f"QX-{r-4}")
        else: labels.append(f"POST-{r-8}")
        
    ax.set_yticklabels(labels, fontweight='bold')
    ax.axhline(y=4.5, color='black', linestyle='-', linewidth=2)
    ax.axhline(y=8.5, color='black', linestyle='-', linewidth=2)
    
    ax.set_xlabel("Tiempo (Minutos)", fontsize=14, fontweight='bold')
    ax.set_title(f"Planificación Quirúrgica Real (CINN + SA) - Makespan: {makespan:.0f} min\nDetalle Visual: Setup (10m) + Cirugía + Cleanup", fontsize=16)
    
    ax.set_xlim(-makespan*0.02, makespan * 1.02)
    ax.set_ylim(0.5, 12.5)
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Gráfico Gantt guardado en: {output_path}")
    plt.close()

#ploteo KPI de histogramas
def plot_wait_histograms(df, output_path="data/processed/esperas_histograma.png"):
    waits_dict = {}
    stages = sorted(df['stage_id'].unique())

    for i in range(len(stages) - 1):
        stage_curr, stage_next = stages[i], stages[i+1]
        delays = []
        for j in df['job_id'].unique():
            row_curr = df[(df['job_id'] == j) & (df['stage_id'] == stage_curr)]
            row_next = df[(df['job_id'] == j) & (df['stage_id'] == stage_next)]
            if row_curr.empty or row_next.empty: continue
            
            end_curr = row_curr['real_end'].values[0]
            start_next = row_next['real_start'].values[0]
            delays.append(start_next - end_curr)
            
        waits_dict[(stage_curr, stage_next)] = delays

    colors = ['#2ca02c', '#1f77b4']
    stage_names = {0: "Preoperatorio", 1: "Quirófano", 2: "Postoperatorio"}

    fig, axes = plt.subplots(1, len(waits_dict), figsize=(12, 5), sharey=True)
    if len(waits_dict) == 1: axes = [axes]

    for idx, ((i, ip1), lst) in enumerate(waits_dict.items()):
        ax = axes[idx]
        arr = np.array(lst, dtype=float)
        ax.hist(arr, bins=10, alpha=0.75, color=colors[idx % len(colors)], edgecolor='black')
        
        mean_val = np.mean(arr)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Media: {mean_val:.1f} min")
        
        ax.set_title(f"Esperas: {stage_names.get(i)} $\\to$ {stage_names.get(ip1)}")
        ax.set_xlabel("Tiempo de Espera (min)")
        if idx == 0: ax.set_ylabel("Frecuencia")
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    plt.suptitle("Distribución de Tiempos Muertos del Paciente", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Histogramas guardados en: {output_path}")
    plt.close()