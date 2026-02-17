import numpy as np
import copy
import math
import random
import torch

def extract_topology(s_pred, m_probs, p_time_medical, p_time_occupancy, J, I, R):
    s_pred_np = s_pred.cpu().numpy()
    p_med_np = p_time_medical.cpu().numpy()
    p_occ_np = p_time_occupancy.cpu().numpy()
    machine_indices = torch.argmax(m_probs, dim=-1).cpu().numpy()

    task_data = []
    for j in range(J):
        for i in range(I):
            local_m = machine_indices[j, i]
            global_m_id = (i * R) + local_m + 1 
            stage_name = ["PRE", "QX", "POST"][i]
            res_name = f"{stage_name}-{local_m+1}"
            
            task_data.append({
                'job_id': j,
                'stage_id': i,
                'pinn_start': s_pred_np[j, i],
                'dur_medical': p_med_np[j, i],
                'dur_occupancy': p_occ_np[j, i],
                'global_machine_id': global_m_id,
                'resource_name': res_name
            })
    return task_data

def calculate_makespan_from_structure(tasks, J):
    machine_avail = {r: 0.0 for r in range(1, 13)} 
    job_avail = {j: 0.0 for j in range(J)}
    tasks.sort(key=lambda x: x['pinn_start'])
    
    final_tasks = []
    for t in tasks:
        j = t['job_id']
        m_global = t['global_machine_id']
        dur_med = t['dur_medical']
        dur_occ = t['dur_occupancy']
        
        start_t = max(job_avail[j], machine_avail[m_global])
        end_t_patient = start_t + dur_med
        end_t_machine = start_t + dur_occ
        
        job_avail[j] = end_t_patient
        machine_avail[m_global] = end_t_machine
        
        new_t = t.copy()
        new_t['real_start'] = start_t
        new_t['real_end'] = end_t_patient
        new_t['pinn_start'] = start_t 
        final_tasks.append(new_t)
        
    makespan = max([t['real_end'] for t in final_tasks])
    return makespan, final_tasks

#Metaheuristica SA
def simulated_annealing_optimization(task_data, J, iterations=5000, initial_temp=100.0, cooling_rate=0.995):
    print("Iniciando Recocido Simulado (SA)...")
    best_tasks = copy.deepcopy(task_data)
    current_tasks = copy.deepcopy(task_data)
    
    best_makespan, best_tasks = calculate_makespan_from_structure(best_tasks, J)
    current_makespan = best_makespan
    
    print(f"Makespan CINN (Topología Inicial): {best_makespan:.2f} min")
    
    machine_ranges = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11, 12]}
    T = initial_temp
    
    for k in range(iterations):
        neighbor_tasks = copy.deepcopy(current_tasks)
        idx = np.random.randint(0, len(neighbor_tasks))
        
        task = neighbor_tasks[idx]
        stage = task['stage_id']
        
        candidates = [m for m in machine_ranges[stage] if m != task['global_machine_id']]
        if not candidates: continue
            
        new_machine = np.random.choice(candidates)
        neighbor_tasks[idx]['global_machine_id'] = new_machine
        
        stage_name = ["PRE", "QX", "POST"][stage]
        neighbor_tasks[idx]['resource_name'] = f"{stage_name}-{new_machine - (stage * 4)}"
        
        neighbor_makespan, updated_schedule = calculate_makespan_from_structure(neighbor_tasks, J)
        
        delta = neighbor_makespan - current_makespan
        
        if delta < 0 or math.exp(-delta / T) > random.random():
            current_tasks = updated_schedule
            current_makespan = neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_tasks = copy.deepcopy(current_tasks)
                
        T = T * cooling_rate

    print(f"Makespan Final Optimizado (SA): {best_makespan:.2f} min")
    return best_tasks

#Algoritmo Hill Climbing
def hill_climbing_optimization(task_data, J, iterations=2000):
    print("Iniciando Búsqueda Local (Hill Climbing)...")
    best_tasks = copy.deepcopy(task_data)
    current_tasks = copy.deepcopy(task_data)
    
    best_makespan, best_tasks = calculate_makespan_from_structure(best_tasks, J)
    current_makespan = best_makespan
    
    print(f"Makespan CINN (Topología Inicial): {best_makespan:.2f} min")
    
    machine_ranges = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11, 12]}
    
    for k in range(iterations):
        # 1. Crear vecino (mutación aleatoria)
        neighbor_tasks = copy.deepcopy(current_tasks)
        idx = np.random.randint(0, len(neighbor_tasks))
        
        task = neighbor_tasks[idx]
        stage = task['stage_id']
        
        candidates = [m for m in machine_ranges[stage] if m != task['global_machine_id']]
        if not candidates: continue
            
        new_machine = np.random.choice(candidates)
        neighbor_tasks[idx]['global_machine_id'] = new_machine
        
        stage_name = ["PRE", "QX", "POST"][stage]
        neighbor_tasks[idx]['resource_name'] = f"{stage_name}-{new_machine - (stage * 4)}"
        
        # Evaluar vecino
        neighbor_makespan, updated_schedule = calculate_makespan_from_structure(neighbor_tasks, J)
        
        # Aceptar SOLO si es MEJOR o IGUAL
        if neighbor_makespan <= current_makespan:
            current_tasks = updated_schedule
            current_makespan = neighbor_makespan
            
            # Actualizar el mejor global encontrado
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_tasks = copy.deepcopy(current_tasks)

    print(f"Makespan Final Optimizado (Hill Climbing): {best_makespan:.2f} min")
    return best_tasks