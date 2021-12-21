import time
import tqdm

from panelObject import Panel

# this line is needed in case we decide to implement multithreading
if __name__ == "__main__":
    # start time
    t0 = time.time()

    # initialize all the panels from the folders
    panels = Panel.initialize_all()

    # shorten the list of panels which should be processed, comment to run through all panels
    panels = panels[:2]

    # for every panel we perform the following actions
    for panel in tqdm.tqdm(panels, desc='Panel'):
        print('\n'+str(panel))
        # panel.load_ae()
        # panel.analyse_ae()

        panel.load_luna()
        panel.analyse_luna()
        panel.plot_luna()

    # end time, it also prints the elapsed time
    t1 = time.time()
    print(f"Total time elapsed: {round((t1-t0)/60,3)} minutes")
