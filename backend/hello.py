import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import lancedb
import pandas as pd
import traceback


class LanceDBViewer:
    def __init__(self, master):
        try:
            print("Initializing LanceDB Viewer")
            self.master = master
            master.title("LanceDB Viewer")
            master.geometry("800x600")

            # Database connection frame
            self.db_frame = ttk.LabelFrame(master, text="Database Connection")
            self.db_frame.pack(padx=10, pady=10, fill="x")

            self.db_path_var = tk.StringVar()
            ttk.Label(self.db_frame, text="LanceDB Path:").pack(
                side="left", padx=5)
            ttk.Entry(self.db_frame, textvariable=self.db_path_var, width=50).pack(
                side="left", padx=5
            )
            ttk.Button(self.db_frame, text="Browse", command=self.browse_db_path).pack(
                side="left"
            )
            ttk.Button(self.db_frame, text="Connect", command=self.connect_db).pack(
                side="left", padx=5
            )

            # Table selection frame
            self.table_frame = ttk.LabelFrame(master, text="Tables")
            self.table_frame.pack(padx=10, pady=10, fill="x")

            self.table_var = tk.StringVar()
            self.table_dropdown = ttk.Combobox(
                self.table_frame, textvariable=self.table_var, state="readonly"
            )
            self.table_dropdown.pack(
                side="left", padx=5, expand=True, fill="x")
            self.table_dropdown.bind(
                "<<ComboboxSelected>>", self.load_table_data)

            # Data view frame
            self.data_frame = ttk.Frame(master)
            self.data_frame.pack(padx=10, pady=10, expand=True, fill="both")

            self.tree = ttk.Treeview(self.data_frame)
            self.tree.pack(side="left", expand=True, fill="both")

            # Scrollbar
            scrollbar = ttk.Scrollbar(
                self.data_frame, orient="vertical", command=self.tree.yview
            )
            scrollbar.pack(side="right", fill="y")
            self.tree.configure(yscroll=scrollbar.set)

            # CRUD Buttons
            self.crud_frame = ttk.Frame(master)
            self.crud_frame.pack(padx=10, pady=10, fill="x")

            ttk.Button(self.crud_frame, text="Create", command=self.create_record).pack(
                side="left", padx=5
            )
            ttk.Button(self.crud_frame, text="Update", command=self.update_record).pack(
                side="left", padx=5
            )
            ttk.Button(self.crud_frame, text="Delete", command=self.delete_record).pack(
                side="left", padx=5
            )

            self.db = None
            self.current_table = None
            print("Initialization complete")
        except Exception as e:
            print(f"Initialization Error: {e}")
            traceback.print_exc()

    def browse_db_path(self):
        try:
            print("Browsing DB Path")
            path = filedialog.askdirectory()
            if path:
                self.db_path_var.set(path)
                print(f"Selected path: {path}")
        except Exception as e:
            print(f"Browse Path Error: {e}")
            traceback.print_exc()

    def connect_db(self):
        try:
            print("Connecting to DB")
            path = self.db_path_var.get()
            print(f"DB Path: {path}")

            if not path:
                messagebox.showerror("Error", "Please select a LanceDB path")
                return

            self.db = lancedb.connect(path)
            tables = self.db.list_tables()
            print(f"Found tables: {tables}")

            self.table_dropdown["values"] = tables
            if tables:
                self.table_dropdown.set(tables[0])
                self.load_table_data()
            else:
                messagebox.showinfo("Info", "No tables found in this database")
        except Exception as e:
            print(f"DB Connection Error: {e}")
            traceback.print_exc()
            messagebox.showerror("Connection Error", str(e))

    def load_table_data(self, event=None):
        try:
            print("Loading table data")
            if not self.db:
                print("No DB connection")
                return

            table_name = self.table_var.get()
            print(f"Selected table: {table_name}")

            if not table_name:
                print("No table selected")
                return

            table = self.db.open_table(table_name)
            df = table.to_pandas()
            print(f"DataFrame shape: {df.shape}")

            # Clear existing tree
            for i in self.tree.get_children():
                self.tree.delete(i)

            # Configure columns
            self.tree["columns"] = list(df.columns)
            self.tree.column("#0", width=0, stretch=tk.NO)

            for col in df.columns:
                self.tree.column(col, anchor=tk.W, width=100)
                self.tree.heading(col, text=col)

            # Add data
            for index, row in df.iterrows():
                self.tree.insert("", "end", values=list(row))

            self.current_table = table
            print("Table data loaded successfully")
        except Exception as e:
            print(f"Load Table Error: {e}")
            traceback.print_exc()
            messagebox.showerror("Load Error", str(e))

    def create_record(self):
        try:
            if not self.current_table:
                messagebox.showerror("Error", "No table selected")
                return

            # Get column names
            columns = self.current_table.to_pandas().columns

            # Create input dialog for each column
            record_data = {}
            for col in columns:
                value = simpledialog.askstring(
                    "Create Record", f"Enter value for {col}:"
                )
                if value is not None:
                    record_data[col] = value

            if record_data:
                self.current_table.add(record_data)
                self.load_table_data()
                messagebox.showinfo("Success", "Record created successfully")
        except Exception as e:
            messagebox.showerror("Create Error", str(e))

    def update_record(self):
        try:
            if not self.current_table:
                messagebox.showerror("Error", "No table selected")
                return

            selected_item = self.tree.selection()
            if not selected_item:
                messagebox.showerror("Error", "Select a record to update")
                return

            # Get current record values
            current_values = self.tree.item(selected_item[0])["values"]
            columns = self.current_table.to_pandas().columns

            # Create update dialog
            record_data = {}
            for col, val in zip(columns, current_values):
                new_value = simpledialog.askstring(
                    "Update Record",
                    f"Update {col} (current: {val})",
                    initialvalue=str(val),
                )
                if new_value is not None:
                    record_data[col] = new_value

            if record_data:
                # Assuming first column is a unique identifier
                primary_key = columns[0]
                self.current_table.update(
                    record_data, f"{primary_key} = '{current_values[0]}'"
                )
                self.load_table_data()
                messagebox.showinfo("Success", "Record updated successfully")
        except Exception as e:
            messagebox.showerror("Update Error", str(e))

    def delete_record(self):
        try:
            if not self.current_table:
                messagebox.showerror("Error", "No table selected")
                return

            selected_item = self.tree.selection()
            if not selected_item:
                messagebox.showerror("Error", "Select a record to delete")
                return

            # Get current record values
            current_values = self.tree.item(selected_item[0])["values"]
            columns = self.current_table.to_pandas().columns

            # Assuming first column is a unique identifier
            primary_key = columns[0]

            # Confirm deletion
            if messagebox.askyesno(
                "Confirm", "Are you sure you want to delete this record?"
            ):
                self.current_table.delete(
                    f"{primary_key} = '{current_values[0]}'")
                self.load_table_data()
                messagebox.showinfo("Success", "Record deleted successfully")
        except Exception as e:
            messagebox.showerror("Delete Error", str(e))


def main():
    try:
        print("Starting LanceDB Viewer")
        root = tk.Tk()
        app = LanceDBViewer(root)
        root.mainloop()
    except Exception as e:
        print(f"Main Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
