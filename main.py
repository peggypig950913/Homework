import tkinter as tk
from ui.app import GestureInstrumentApp

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = GestureInstrumentApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
