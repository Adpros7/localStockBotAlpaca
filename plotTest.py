import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [10, 20, 30]
info = {
    0: "First point details",
    1: "Second point details",
    2: "Third point details",
}

fig, ax = plt.subplots()
scatter = ax.scatter(x, y, picker=True)


def on_pick(event):
    idx = event.ind[0]
    print(info[idx])


fig.canvas.mpl_connect("pick_event", on_pick)

plt.show()
