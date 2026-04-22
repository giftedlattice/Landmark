import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class RoomReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Room Reconstruction Tool")

        self.df = None
        self.points = {}              # {point_name: np.array([x,y,z])}
        self.point_names = []         # ordered list of point names
        self.point_coords = None      # Nx3 array
        self.assigned_points = set()  # point names already assigned to objects
        self.selected_points = []     # currently selected points
        self.objects = {}             # {object_name: [point_names]}
        self.object_colors = {}       # {object_name: color}

        self.selection_armed = False
        self.current_plane = "xy"

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        tk.Button(control_frame, text="Load Export CSV", command=self.load_csv, width=22).pack(pady=4)
        tk.Button(control_frame, text="Arm Selection", command=self.arm_selection, width=22).pack(pady=4)
        tk.Button(control_frame, text="Disarm Selection", command=self.disarm_selection, width=22).pack(pady=4)
        tk.Button(control_frame, text="Clear Current Selection", command=self.clear_selection, width=22).pack(pady=4)
        tk.Button(control_frame, text="Create Object", command=self.create_object, width=22).pack(pady=4)
        tk.Button(control_frame, text="Delete Object", command=self.delete_object, width=22).pack(pady=4)
        tk.Button(control_frame, text="Save Objects JSON", command=self.save_objects_json, width=22).pack(pady=4)

        tk.Label(control_frame, text="View Plane").pack(pady=(12, 2))
        self.plane_var = tk.StringVar(value="xy")
        plane_menu = tk.OptionMenu(control_frame, self.plane_var, "xy", "xz", "yz", command=self.change_plane)
        plane_menu.config(width=18)
        plane_menu.pack(pady=4)

        tk.Label(control_frame, text="Objects").pack(pady=(12, 2))
        self.object_listbox = tk.Listbox(control_frame, width=28, height=18)
        self.object_listbox.pack(pady=4)

        tk.Label(control_frame, text="Status").pack(pady=(12, 2))
        self.status_var = tk.StringVar(value="Load a CSV to begin.")
        tk.Label(control_frame, textvariable=self.status_var, wraplength=180, justify="left").pack()

    def _build_plot(self):
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select export CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            self.points = self.extract_points_from_export(self.df)
            self.point_names = sorted(self.points.keys())
            self.point_coords = np.array([self.points[name] for name in self.point_names], dtype=float)

            self.assigned_points = set()
            self.selected_points = []
            self.objects = {}
            self.object_colors = {}
            self.refresh_object_list()
            self.redraw_plot()

            self.status_var.set(f"Loaded {len(self.point_names)} points from {Path(file_path).name}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load CSV:\n{e}")

    @staticmethod
    def extract_points_from_export(df):
        """
        Expects columns like:
        name_x, name_y, name_z

        Uses the mean across rows for each named point.
        """
        cols = list(df.columns)
        base_names = []

        for col in cols:
            if col.endswith("_x"):
                base = col[:-2]
                if f"{base}_y" in cols and f"{base}_z" in cols:
                    base_names.append(base)

        if not base_names:
            raise ValueError("No point columns found. Expected columns ending in _x, _y, _z")

        points = {}
        for base in base_names:
            x = pd.to_numeric(df[f"{base}_x"], errors="coerce").mean()
            y = pd.to_numeric(df[f"{base}_y"], errors="coerce").mean()
            z = pd.to_numeric(df[f"{base}_z"], errors="coerce").mean()
            points[base] = np.array([x, y, z], dtype=float)

        return points

    def change_plane(self, value):
        self.current_plane = value
        self.redraw_plot()

    def arm_selection(self):
        self.selection_armed = True
        self.status_var.set("Selection armed. Click points to select them.")

    def disarm_selection(self):
        self.selection_armed = False
        self.status_var.set("Selection disarmed.")

    def clear_selection(self):
        self.selected_points = []
        self.redraw_plot()
        self.status_var.set("Current selection cleared.")

    def create_object(self):
        if not self.selected_points:
            messagebox.showinfo("No Selection", "No points selected.")
            return

        object_name = simpledialog.askstring("Object Name", "Enter a name for this object:")
        if not object_name:
            return

        if object_name in self.objects:
            messagebox.showerror("Duplicate Name", "An object with that name already exists.")
            return

        self.objects[object_name] = list(self.selected_points)
        self.assigned_points.update(self.selected_points)
        self.object_colors[object_name] = self.get_next_color(len(self.objects) - 1)

        self.selected_points = []
        self.refresh_object_list()
        self.redraw_plot()
        self.status_var.set(f"Created object '{object_name}'.")

    def delete_object(self):
        selection = self.object_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Object Selected", "Select an object in the list first.")
            return

        object_name = self.object_listbox.get(selection[0])

        point_names = self.objects.pop(object_name, [])
        self.object_colors.pop(object_name, None)

        # rebuild assigned points from remaining objects
        self.assigned_points = set()
        for pts in self.objects.values():
            self.assigned_points.update(pts)

        self.refresh_object_list()
        self.redraw_plot()
        self.status_var.set(f"Deleted object '{object_name}'.")

    def save_objects_json(self):
        if not self.objects:
            messagebox.showinfo("Nothing to Save", "No objects have been created yet.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save objects JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not save_path:
            return

        output = {
            "plane_view_last_used": self.current_plane,
            "objects": {}
        }

        for object_name, point_names in self.objects.items():
            output["objects"][object_name] = {
                "points": point_names,
                "coordinates": {
                    name: self.points[name].tolist()
                    for name in point_names
                }
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        self.status_var.set(f"Saved objects to {Path(save_path).name}")
        messagebox.showinfo("Saved", f"Objects saved to:\n{save_path}")

    def refresh_object_list(self):
        self.object_listbox.delete(0, tk.END)
        for obj_name in self.objects:
            self.object_listbox.insert(tk.END, obj_name)

    def get_plane_indices(self):
        if self.current_plane == "xy":
            return 0, 1, "X", "Y"
        elif self.current_plane == "xz":
            return 0, 2, "X", "Z"
        elif self.current_plane == "yz":
            return 1, 2, "Y", "Z"
        raise ValueError("Invalid plane")

    def redraw_plot(self):
        self.ax.clear()

        if self.point_coords is None or len(self.point_coords) == 0:
            self.ax.set_title("No data loaded")
            self.canvas.draw()
            return

        i, j, xlabel, ylabel = self.get_plane_indices()

        # unassigned points
        unassigned_names = [name for name in self.point_names if name not in self.assigned_points]
        if unassigned_names:
            coords = np.array([self.points[name] for name in unassigned_names])
            self.ax.scatter(coords[:, i], coords[:, j], s=40, label="unassigned")

        # assigned objects
        for obj_name, point_names in self.objects.items():
            coords = np.array([self.points[name] for name in point_names])
            color = self.object_colors.get(obj_name, None)
            self.ax.scatter(coords[:, i], coords[:, j], s=70, label=obj_name, color=color)

            if len(coords) > 1:
                self.ax.plot(coords[:, i], coords[:, j], color=color, alpha=0.8)

        # current selection
        if self.selected_points:
            coords = np.array([self.points[name] for name in self.selected_points])
            self.ax.scatter(
                coords[:, i], coords[:, j],
                s=120, marker="o", facecolors="none", edgecolors="red",
                linewidths=2, label="current selection"
            )

        # labels
        for name in self.point_names:
            coord = self.points[name]
            self.ax.text(coord[i], coord[j], name, fontsize=8)

        self.ax.set_title(f"Room Reconstruction ({self.current_plane.upper()})")
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend(fontsize=8, loc="best")
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def on_plot_click(self, event):
        if not self.selection_armed:
            return
        if event.inaxes != self.ax:
            return
        if self.point_coords is None or len(self.point_coords) == 0:
            return
        if event.xdata is None or event.ydata is None:
            return

        i, j, _, _ = self.get_plane_indices()

        click_xy = np.array([event.xdata, event.ydata])

        min_dist = None
        nearest_name = None

        for name in self.point_names:
            coord = self.points[name]
            point_2d = np.array([coord[i], coord[j]])
            dist = np.linalg.norm(point_2d - click_xy)

            if min_dist is None or dist < min_dist:
                min_dist = dist
                nearest_name = name

        if nearest_name is None:
            return

        # Toggle select
        if nearest_name in self.selected_points:
            self.selected_points.remove(nearest_name)
            self.status_var.set(f"Removed '{nearest_name}' from current selection.")
        else:
            if nearest_name in self.assigned_points:
                self.status_var.set(f"'{nearest_name}' already belongs to an object.")
                return
            self.selected_points.append(nearest_name)
            self.status_var.set(f"Selected '{nearest_name}'.")

        self.redraw_plot()

    @staticmethod
    def get_next_color(index):
        cmap = plt.get_cmap("tab20")
        return cmap(index % 20)


if __name__ == "__main__":
    root = tk.Tk()
    app = RoomReconstructionApp(root)
    root.geometry("1400x900")
    root.mainloop()