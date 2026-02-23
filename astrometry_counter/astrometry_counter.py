from datetime import datetime

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
        'opposition_count': 0
    }
    
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
            "opp_with_second_most_nights": get_top_n(all_nights)[1]
        }
    
    def reset_tracking():
        """Reset tracking variables for new object."""
        state.update({
            'times': {'recent': None, 'first_ever': None, 'first_opp': None, 'last_night': None},
            'arcs': {'day': 0, 'gap': 0, 'opp': 0},
            'lists': {'day_arcs': [], 'gap_arcs': [], 'opp_arcs': [], 'opp_gaps': []},
            'nights': {'total': 0, 'current_opp': 0, 'per_opp': []},
            'opposition_count': 1
        })

    for line in lines:
        packed_desig = line[5:12]
        try:
            d = datetime.strptime(line[15:25], "%Y %m %d").toordinal() + float(line[25:32])
        except (ValueError, IndexError):
            continue
        
        # New object detected
        if packed_desig != current_desig:
            save_object()
            current_desig = packed_desig
            reset_tracking()
            state['times'].update({'recent': d, 'first_ever': d, 'first_opp': d, 'last_night': d})
            state['nights'].update({'total': 1, 'current_opp': 1})
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
    
    save_object()
    return objects