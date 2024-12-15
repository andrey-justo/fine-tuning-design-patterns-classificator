import os
from enum import Enum
from pathlib import Path, PureWindowsPath

class Category(str, Enum):
    Unknown = 'unknown'
    FeatureFlag = 'feature_flags'

class Language(str, Enum):
    Java = 'java'
    Kotlin = 'kotlin'
    Golang = 'golang'
    Javascript = 'javascript'
    CSharp = 'csharp'
    Python = 'python'
    Exilir = 'exilir'
    Ruby = 'ruby'
    CPluPlus = 'c++'
    Unknown = 'unknown'

    def extract_from_file_extension(path):
        _, file_extension = os.path.splitext(path)
        file_extension = file_extension.replace(".", "")
        match file_extension:
            case 'js': return Language.Javascript
            case 'java': return Language.Java
            case 'kt': return Language.Kotlin
            case 'cs': return Language.CSharp
            case 'py': return Language.Python
            case 'ex' | 'exs': return Language.Exilir
            case 'ruby': return Language.Ruby
            case 'cpp': return Language.CPluPlus
            case 'go': return Language.Golang
            case _: return Language.Unknown


class DesignPatternDataRow:
    ## TODO: Create validate function for logs
    def __init__(self, content_path, category, reference, language=None):
        self._content_path = content_path
        if not language:
            self._language = Language.extract_from_file_extension(content_path)
        else:
            self._language = language

        if not category:
            self._category = self.extract_from_folder(content_path)
        else:
            self._category = category
        self._reference = reference
        self._content = self.read_file(content_path)

    def read_file(self, path):
        f = open(path, "r")
        return f.read()

    def extract_from_folder(self, path):
        if not path:
            return Category.Unknown
        
        ## FIXME: normalize splitter
        parts = PureWindowsPath(path).parts
        next_category_folder = False
        for i in range(0, len(parts)):
            if next_category_folder:
                return parts[i]
            
            if parts[i] == 'samples':
                next_category_folder = True

        return Category.Unknown
        