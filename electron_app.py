import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

app = QApplication(sys.argv)

view = QWebEngineView()
view.load(QUrl("http://localhost:5000"))
view.show()

sys.exit(app.exec_())
