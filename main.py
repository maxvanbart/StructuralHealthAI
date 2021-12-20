import time

from panelObject import Panel

# this line is needed in case we decide to implement multithreading
if __name__ == "__main__":
    # start time
    t0 = time.time()

    # initialize all the panels from the folders
    panels = Panel.initialize_all()

    # for every array we perform the following actions
    for panel in panels:
        print('\n'+str(panel))

        panel.load_luna()
        panel.analyse_luna()
        panel.plot_luna()

    # end time, it also prints the elapsed time
    t1 = time.time()

    print(f"Total time elapsed: {round(t1-t0,3)} seconds")
