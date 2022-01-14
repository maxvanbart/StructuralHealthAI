import time
from tqdm import tqdm
from panelObject import Panel

# This line is needed for implementing multi-threading
if __name__ == "__main__":
    # start time
    t0 = time.time()

    # initialize all the panels from the folders
    panels = Panel.initialize_all(force_clustering=True)

    # for every panel we perform the following actions
    for panel in tqdm(panels, desc='Panel'):
        print('\n'+str(panel))
        # Prepare AE.
        panel.load_ae()
        panel.analyse_ae()

        # Prepare LUNA.
        panel.load_luna()
        panel.synchronise_luna()
        panel.analyse_luna()

        # Do PZT stuff
        panel.load_pzt()
        panel.synchronise_pzt()
        panel.analyse_pzt()

        # Plot and save all the clusters.
        panel.visualize_all()
        panel.save_all()

    # end time, it also prints the elapsed time
    t1 = time.time()
    print(f"Total time elapsed: {round((t1-t0)/60,3)} minutes")
