from datetime import datetime

V_BAND_CORRECTIONS = {
    ' ': -0.8,
    'U': -1.3, 'B': -0.8, 'g': -0.35, 'V': 0.0, 'r': 0.14,
    'R': 0.4, 'C': 0.4, 'W': 0.4, 'i': 0.32, 'z': 0.26,
    'I': 0.8, 'J': 1.2, 'w': -0.13, 'y': 0.32, 'L': 0.2,
    'H': 1.4, 'K': 1.7, 'Y': 0.7, 'G': 0.28, 'v': 0.0,
    'c': -0.05, 'o': 0.33, 'u': 2.5,
}


def process_astrometry(lines):
    """
    Process astronomical astrometry observations to calculate temporal metrics for each object.
    
    Calculates "day arcs" (consecutive observations within 0.5 days) and "gap arcs" 
    (periods between observing sessions ≥0.5 days apart) for objects in MPC80 format.
    Also tracks oppositional arcs and gaps (180+ day separations) for long-term monitoring.
    
    Args:
        lines: List of observation lines in MPC80 fixed-width format
        
    Returns:
        Dictionary mapping object designations to their temporal metrics
    """
    SAME_DAY_THRESHOLD, OPPOSITION_GAP_THRESHOLD = 0.5, 180.0
    
    lines = sorted(lines, key=lambda line: line[5:12] + line[15:32])
    objects, current_desig = {}, ""
    
    # State dictionary eliminates need for many nonlocal declarations
    state = {
        'times': {'recent': None, 'first_ever': None, 'first_opp': None, 'last_night': None},
        'arcs': {'day': 0, 'gap': 0, 'opp': 0},
        'lists': {'day_arcs': [], 'gap_arcs': [], 'opp_arcs': [], 'opp_gaps': []},
        'nights': {'total': 0, 'current_opp': 0, 'per_opp': []},
        'opposition_count': 0,
        'v_mags': []
    }

    def extract_v_mag(line):
        """Return V-band-corrected magnitude from MPC80 line, or None if unparseable."""
        if len(line) < 71:
            return None
        try:
            mag = float(line[65:70])
        except ValueError:
            return None
        return mag + V_BAND_CORRECTIONS.get(line[70], 0.0)
    
    def get_top_n(lst, n=2, reverse=True):
        """Get top n elements from list, padded with zeros."""
        sorted_list = sorted(lst, reverse=reverse)
        return [sorted_list[i] if i < len(sorted_list) else 0 for i in range(n)]
    
    def save_object():
        """Save current object's metrics to results dict."""
        if not current_desig:
            return
        
        s = state  # Shorthand for readability
        opp_arcs = [a for a in s['lists']['opp_arcs'] + ([s['arcs']['opp']] if s['arcs']['opp'] > 0 else []) if a > 0]
        opp_gaps = [g for g in s['lists']['opp_gaps'] if g > 0]
        all_nights = s['nights']['per_opp'] + ([s['nights']['current_opp']] if s['nights']['current_opp'] > 0 else [])
        v_mags = s['v_mags']
        v_mags_sorted = sorted(v_mags) if v_mags else []

        objects[current_desig] = {
            "longest_day_arc": max(s['lists']['day_arcs'], default=0),
            "longest_gap_arc": max(s['lists']['gap_arcs'], default=0),
            "second_longest_gap_arc": get_top_n(s['lists']['gap_arcs'])[1],
            "shortest_gap_arc": min(s['lists']['gap_arcs'], default=0),
            "total_arc": (s['times']['recent'] - s['times']['first_ever']) if s['times']['first_ever'] else 0,
            "longest_opp_arc": max(opp_arcs, default=0),
            "shortest_opp_arc": min(opp_arcs, default=0),
            "longest_opp_gap": max(opp_gaps, default=0),
            "shortest_opp_gap": min(opp_gaps, default=0),
            "opposition_count": s['opposition_count'],
            "nights_total": s['nights']['total'],
            "opp_with_most_nights": get_top_n(all_nights)[0],
            "opp_with_second_most_nights": get_top_n(all_nights)[1],
            "v_mag_min": v_mags_sorted[0] if v_mags_sorted else None,
            "v_mag_second_min": v_mags_sorted[1] if len(v_mags_sorted) > 1 else None,
            "v_mag_max": v_mags_sorted[-1] if v_mags_sorted else None,
            "v_mag_second_max": v_mags_sorted[-2] if len(v_mags_sorted) > 1 else None,
            "v_mag_avg": sum(v_mags) / len(v_mags) if v_mags else None,
        }
    
    def reset_tracking():
        """Reset tracking variables for new object."""
        state.update({
            'times': {'recent': None, 'first_ever': None, 'first_opp': None, 'last_night': None},
            'arcs': {'day': 0, 'gap': 0, 'opp': 0},
            'lists': {'day_arcs': [], 'gap_arcs': [], 'opp_arcs': [], 'opp_gaps': []},
            'nights': {'total': 0, 'current_opp': 0, 'per_opp': []},
            'opposition_count': 1,
            'v_mags': []
        })

    previous_id_time = ""

    for line in lines:
        # these four lines filter out roving observers and satellites (the second line of each)
        id_time = line[5:12] + line[15:32]
        if id_time == previous_id_time:
            continue
        previous_id_time = id_time

        packed_desig = line[5:12]
        try:
            d = datetime.strptime(line[15:25], "%Y %m %d").toordinal() + float(line[25:32])
        except (ValueError, IndexError):
            continue
        
        v_mag = extract_v_mag(line)

        # New object detected
        if packed_desig != current_desig:
            save_object()
            current_desig = packed_desig
            reset_tracking()
            state['times'].update({'recent': d, 'first_ever': d, 'first_opp': d, 'last_night': d})
            state['nights'].update({'total': 1, 'current_opp': 1})
            if v_mag is not None:
                state['v_mags'].append(v_mag)
            continue
        
        time_gap = d - state['times']['recent']
        
        # New opposition detected
        if time_gap >= OPPOSITION_GAP_THRESHOLD:
            if state['arcs']['opp'] > 0:
                state['lists']['opp_arcs'].append(state['arcs']['opp'])
            state['lists']['opp_gaps'].append(time_gap)
            state['nights']['per_opp'].append(state['nights']['current_opp'])
            state['opposition_count'] += 1
            state['times'].update({'first_opp': d})
            state['arcs']['opp'] = 0
            state['nights']['current_opp'] = 0
        
        # Track new night (gap > 0.5 days from last night start)
        if state['times']['last_night'] and (d - state['times']['last_night']) > SAME_DAY_THRESHOLD:
            state['nights']['total'] += 1
            state['nights']['current_opp'] += 1
            state['times']['last_night'] = d
        
        state['arcs']['opp'] = d - state['times']['first_opp']
        
        # Handle day arc vs gap arc
        if time_gap < SAME_DAY_THRESHOLD:
            state['arcs']['day'] += time_gap
            if state['arcs']['gap'] > 0:
                state['lists']['gap_arcs'].append(state['arcs']['gap'])
                state['arcs']['gap'] = 0
        else:
            state['arcs']['gap'] += time_gap
            if state['arcs']['day'] > 0:
                state['lists']['day_arcs'].append(state['arcs']['day'])
                state['arcs']['day'] = 0
        
        state['times']['recent'] = d

        if v_mag is not None:
            state['v_mags'].append(v_mag)

    save_object()
    return objects