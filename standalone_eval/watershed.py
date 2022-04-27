

def watershed(windows, tau, gamma):
    """
    run 1D watershed algo on windows
    
    input:
        windows: [[1.2, 15.3, 0.32], [15.3, 18.6, 0.34], ....]
        segments of [start_time, end_time, similarity_score]
    
    output:
        windows: [[1.2, 18.6, 0.33], ...]
        merged segments
    """
    
    