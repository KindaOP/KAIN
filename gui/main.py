import wx


class MainFrame(wx.Frame):
    def __init__(self, title:str):
        super().__init__(
            None, 
            wx.ID_ANY, 
            title, 
            wx.DefaultPosition, 
            wx.DefaultSize, 
            wx.DEFAULT_FRAME_STYLE|wx.MAXIMIZE_BOX
        )
        self.Bind(wx.EVT_CLOSE, self.OnClose, self)

        mainSizer = wx.BoxSizer(wx.VERTICAL)

        # Menu bar
        menuBar = wx.MenuBar()
        fileMenu = wx.Menu()
        m_new = fileMenu.Append(wx.ID_NEW, "&New\tCtrl+N", "Create a new file.")
        self.Bind(wx.EVT_MENU, self.OnNew, m_new) 
        fileMenu.AppendSeparator()
        m_open = fileMenu.Append(wx.ID_ANY, "&Open\tCtrl+O", "Open an existing file.")
        self.Bind(wx.EVT_MENU, self.OnOpen, m_open)
        fileMenu.AppendSeparator()
        m_exit = fileMenu.Append(wx.ID_EXIT, "&Exit\tCtrl+Q", "Exit the program.")
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)
        menuBar.Append(fileMenu, "&File")
        aboutMenu = wx.Menu()
        m_about = aboutMenu.Append(wx.ID_ABOUT, "&About\tCtrl+A", "Show the About dialog")
        self.Bind(wx.EVT_MENU, self.OnAbout, m_about)
        menuBar.Append(aboutMenu, "&About")
        self.SetMenuBar(menuBar)

        mainNotebook = wx.Notebook(self)
        mainNotebookSizer = wx.BoxSizer(wx.HORIZONTAL)
        verifPanel = wx.Panel(mainNotebook)
        mainNotebook.AddPage(verifPanel, "Verification")
        resultPanel = wx.Panel(mainNotebook)
        mainNotebook.AddPage(resultPanel, "Result")
        mainNotebook.SetSizer(mainNotebookSizer)
        mainSizer.Add(mainNotebook, 1, wx.EXPAND)

        self.SetSizer(mainSizer)

        self.SetInitialSize(wx.Size(800, 600))
        self.Layout()

    def OnClose(self, event:wx.CloseEvent):
        self.Destroy()

    def OnNew(self, event:wx.MenuEvent):
        pass

    def OnOpen(self, event:wx.MenuEvent):
        pass

    def OnAbout(self, event:wx.MenuEvent):
        pass


class MainApp(wx.App):
    def __init__(self):
        super().__init__()

    def OnInit(self) -> bool:
        mainFrame = MainFrame("KAIN")
        mainFrame.Show(True)
        return True