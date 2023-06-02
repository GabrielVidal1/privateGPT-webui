"""Loader that loads Markdown files."""
from typing import List, Union, Any

from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

class MarkdownLoader(UnstructuredMarkdownLoader):
    """Loader that uses unstructured to load markdown files."""

    def __init__(
        self,
        file_path: Union[str, List[str]],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """Initialize with file path."""
        # if file_path is a string, get the title of the markdown file
        if isinstance(file_path, str):
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("# "):
                        self.title = line.replace("# ", "").strip()
                        break

        super().__init__(file_path, mode=mode, **unstructured_kwargs)
    
    def _get_metadata(self) -> dict:
        return {
            "source": self.file_path,
            "title": self.title,
        }