"""Public WinForms folder picker for the bundled pywebview/CoreCLR host."""


class OpenFolderDialog:
    @classmethod
    def show(cls,parent=None,initialDirectory=None,allow_multiple=False,title=None):
        from System.Windows.Forms import DialogResult,FolderBrowserDialog
        dialog=FolderBrowserDialog()
        try:
            dialog.Description=title or ''
            dialog.UseDescriptionForTitle=True
            dialog.SelectedPath=initialDirectory or ''
            dialog.Multiselect=bool(allow_multiple)
            result=dialog.ShowDialog(parent) if parent is not None else dialog.ShowDialog()
            if result!=DialogResult.OK:return None
            return tuple(dialog.SelectedPaths) if allow_multiple else (str(dialog.SelectedPath),)
        finally:
            dialog.Dispose()
