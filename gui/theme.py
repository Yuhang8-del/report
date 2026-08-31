"""Professional visual language for the Fruit SSOD desktop application."""

from __future__ import annotations


APP_STYLE = r"""
* {
    font-family: "Segoe UI", "Arial", sans-serif;
    color: #183247;
}

QMainWindow, QWidget#mainWindowContent {
    background: #eaf0f5;
}

QToolTip {
    color: #ffffff;
    background: #163448;
    border: 1px solid #2a586f;
    padding: 6px 8px;
}

QFrame#topHeader {
    background: #0b2b3d;
    border: 1px solid #18465d;
    border-radius: 16px;
}

QLabel#brandTitle {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
}

QLabel#brandSubtitle, QLabel#modelHint {
    color: #bcd1dc;
    font-size: 12px;
}

QLabel#headerBadge {
    color: #c8f3e8;
    background: #164b50;
    border: 1px solid #267169;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
}

QFrame#navigationCard {
    background: #f9fbfc;
    border: 1px solid #d6e1e8;
    border-radius: 14px;
}

QLabel#navigationTitle {
    color: #122e42;
    font-size: 16px;
    font-weight: 700;
    padding-left: 8px;
}

QLabel#navigationCaption {
    color: #78909f;
    font-size: 11px;
    padding: 0 0 5px 8px;
}

QLabel#navigationFooter {
    color: #547081;
    background: #edf4f6;
    border: 1px solid #d9e6ea;
    border-radius: 10px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.4;
}

QListWidget#navigationList {
    background: transparent;
    border: none;
    outline: none;
    padding: 2px;
}

QListWidget#navigationList::item {
    color: #4d6676;
    border-radius: 9px;
    padding: 13px 14px;
    margin: 3px 0;
    font-size: 14px;
}

QListWidget#navigationList::item:hover {
    background: #e7f1f2;
    color: #153d4e;
}

QListWidget#navigationList::item:selected {
    background: #d9efec;
    color: #12675d;
    font-weight: 700;
    border-left: 4px solid #1d9b89;
}

QPushButton {
    min-height: 20px;
    color: #234357;
    background: #ffffff;
    border: 1px solid #c5d3dc;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
}

QPushButton:hover {
    color: #0e6359;
    background: #eef9f7;
    border-color: #2f9e8f;
}

QPushButton:pressed {
    background: #d9efeb;
}

QPushButton:disabled {
    color: #94a4af;
    background: #eef2f4;
    border-color: #d8e0e5;
}

QPushButton[primary="true"], QPushButton#loadModelButton {
    color: #ffffff;
    background: #168779;
    border-color: #168779;
}

QPushButton[primary="true"]:hover, QPushButton#loadModelButton:hover {
    color: #ffffff;
    background: #119987;
    border-color: #22ab99;
}

QPushButton#releaseModelButton {
    color: #d9e9ef;
    background: transparent;
    border-color: #55798b;
}

QPushButton#releaseModelButton:hover {
    color: #ffffff;
    background: #173f53;
}

QFrame#pageCard, QFrame#infoCard, QFrame#metricCard,
QFrame#cameraControlPanel, QFrame#liveMetricCard {
    background: #ffffff;
    border: 1px solid #d8e3ea;
    border-radius: 13px;
}

QWidget#cameraPage, QWidget#experimentInfoPage {
    background: transparent;
}

QLabel#pageTitle {
    color: #13384f;
    font-size: 20px;
    font-weight: 700;
}

QLabel#pageIntro, QLabel#infoBody, QLabel#statusMessageLabel,
QLabel#imageInferenceSummary, QLabel#videoInferenceSummary,
QLabel#cameraSummary, QLabel#cameraBoundaryNote {
    color: #5c7484;
    font-size: 12px;
}

QLabel#cameraBoundaryNote {
    color: #657b88;
    background: #f3f7f8;
    border: 1px solid #dce7eb;
    border-radius: 8px;
    padding: 9px 11px;
}

QLabel#metricValue {
    color: #123d52;
    font-size: 22px;
    font-weight: 700;
}

QLabel#metricCaption, QLabel#liveMetricCaption {
    color: #728795;
    font-size: 11px;
}

QLabel#liveMetricValue {
    color: #0e6f64;
    font-size: 20px;
    font-weight: 700;
}

QLabel#cameraStateBadge {
    color: #126b5d;
    background: #dff3ee;
    border: 1px solid #b8e2d9;
    border-radius: 10px;
    padding: 7px 11px;
    font-size: 12px;
    font-weight: 700;
}

QLabel#chip, QLabel#extendedChip {
    background: #e7f5f2;
    color: #12675d;
    border: 1px solid #cde9e3;
    border-radius: 8px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#extendedChip {
    color: #5b4b8a;
    background: #f0edfa;
    border-color: #ddd6f3;
}

QLabel#chipLabel {
    color: #536c7b;
    font-size: 11px;
    font-weight: 700;
}

QGroupBox {
    color: #31556b;
    background: #ffffff;
    border: 1px solid #d8e3ea;
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 700;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background: #ffffff;
}

QComboBox, QDoubleSpinBox, QSpinBox {
    min-height: 22px;
    color: #244457;
    background: #f8fafb;
    border: 1px solid #c8d6df;
    border-radius: 7px;
    padding: 6px 9px;
}

QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
    border-color: #5aaea2;
}

QComboBox::drop-down {
    width: 24px;
    border: none;
}

QComboBox QAbstractItemView {
    color: #213e50;
    background: #ffffff;
    border: 1px solid #c8d6df;
    selection-color: #0d5f56;
    selection-background-color: #dff1ee;
}

QProgressBar {
    color: #4f6978;
    background: #e9eff2;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
}

QProgressBar::chunk {
    background: #1d9b89;
    border-radius: 5px;
}

QLabel#imagePreview, QLabel#videoFramePreview, QWidget#cameraFramePreview {
    background: #102938;
    border: 1px solid #285065;
    border-radius: 12px;
}

QFrame#cameraPreviewCard {
    background: #0c2230;
    border: 1px solid #244a5d;
    border-radius: 14px;
}

QWidget#statusPanel {
    background: #f7fafb;
    border: 1px solid #d5e0e6;
    border-radius: 10px;
}

QLabel#modelStateLabel {
    color: #12675d;
    font-size: 12px;
    font-weight: 700;
}

QLabel#singleFileLabel, QLabel#batchFileLabel, QLabel#videoFileLabel, QLabel#videoFpsLabel {
    color: #496779;
    font-size: 12px;
}

QScrollBar:vertical {
    width: 10px;
    margin: 2px;
    background: transparent;
}

QScrollBar::handle:vertical {
    min-height: 30px;
    background: #bfced6;
    border-radius: 5px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
