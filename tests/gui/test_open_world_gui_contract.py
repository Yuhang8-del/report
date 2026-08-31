"""GUI boundary tests for the deliberately disabled open-world extension."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabBar,
    QTextEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from fruit_ssod.gui.main_window import MainWindow


_OPEN_WORLD_TERMS = (
    "open-world",
    "open world",
    "\u5f00\u653e\u4e16\u754c",
    "unknown fruit",
    "unknown",
    "\u672a\u77e5\u6c34\u679c",
    "\u672a\u77e5",
    "discovery",
    "\u53d1\u73b0",
    "novelty",
    "\u65b0\u7c7b\u522b",
    "registry",
    "\u7c7b\u522b\u6ce8\u518c",
)
_VISIBLE_TEXT_ACCESSORS = ("text", "toolTip", "statusTip", "accessibleName", "objectName")
_INTERACTIVE_WIDGET_TYPES = (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTabBar,
    QTextEdit,
)


def _safe_control_text(control: QObject, accessor_name: str) -> str:
    """Read optional Qt metadata without failing on deleted or partial objects."""
    try:
        accessor = getattr(control, accessor_name)
        value = accessor() if callable(accessor) else ""
    except (AttributeError, RuntimeError, TypeError):
        return ""
    return value if isinstance(value, str) else ""


def _advertises_open_world(control: QObject) -> bool:
    """Detect deferred-functionality language in visible or accessible metadata.

    Deliberately do not treat generic terms such as ``class`` as a match: the
    current fixed five-class detector legitimately uses that word throughout
    its conventional UI.
    """
    return any(
        _text_advertises_open_world(_safe_control_text(control, accessor_name))
        for accessor_name in _VISIBLE_TEXT_ACCESSORS
    )


def _text_advertises_open_world(text: str) -> bool:
    """Apply the deferred-workflow vocabulary consistently to list entries."""
    return any(term in text.casefold() for term in _OPEN_WORLD_TERMS)


def _list_item_descriptions(root: QObject) -> tuple[tuple[str, bool, str], ...]:
    """Read visible list-entry labels and their individual enabled flags.

    A ``QListWidgetItem`` is not a QObject, so the normal descendant scan
    cannot see a future navigation item such as "Unknown fruit discovery". Plain
    ``QListView`` instances can hold the same affordance in a model instead,
    therefore inspect their display values and flags as well.  ``QListWidget``
    is handled through its concrete items to preserve its exact item state.
    """
    views = list(root.findChildren(QListView))
    if isinstance(root, QListView):
        views.insert(0, root)

    descriptions: list[tuple[str, bool, str]] = []
    for view in views:
        if isinstance(view, QListWidget):
            for row in range(view.count()):
                item = view.item(row)
                if item is None:
                    continue
                descriptions.append(
                    (
                        item.text(),
                        bool(item.flags() & Qt.ItemFlag.ItemIsEnabled),
                        f"{type(view).__name__}[{row}]",
                    )
                )
            continue

        model = view.model()
        if model is None:
            continue
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if not index.isValid():
                continue
            value = model.data(index, Qt.ItemDataRole.DisplayRole)
            text = value if isinstance(value, str) else ""
            descriptions.append(
                (
                    text,
                    bool(model.flags(index) & Qt.ItemFlag.ItemIsEnabled),
                    f"{type(view).__name__}[{row}]",
                )
            )
    return tuple(descriptions)


def _interactive_descendants(root: QObject) -> tuple[QObject, ...]:
    """Return UI controls that could make an unsupported workflow executable."""
    return tuple(
        child
        for child in root.findChildren(QObject)
        if isinstance(child, (QAction, *_INTERACTIVE_WIDGET_TYPES))
    )


def _describe_control(control: QObject) -> str:
    details = ", ".join(
        f"{accessor_name}={_safe_control_text(control, accessor_name)!r}"
        for accessor_name in _VISIBLE_TEXT_ACCESSORS
    )
    return f"{type(control).__name__}({details})"


def _assert_advertised_controls_are_disabled(root: QObject) -> None:
    control_offenders = [
        control
        for control in _interactive_descendants(root)
        if _advertises_open_world(control) and control.isEnabled()
    ]
    list_item_offenders = [
        f"{source}(text={text!r}, enabled={enabled!r})"
        for text, enabled, source in _list_item_descriptions(root)
        if _text_advertises_open_world(text) and enabled
    ]
    offenders = [*(_describe_control(control) for control in control_offenders), *list_item_offenders]
    assert not offenders, (
        "Open-world controls must remain disabled until the reserved extension is delivered: "
        + "; ".join(offenders)
    )


def test_gui_has_no_enabled_open_world_control_or_action(qtbot: object) -> None:
    """All discoverable open-world affordances in the shipped GUI are disabled."""
    window = MainWindow()
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    _assert_advertised_controls_are_disabled(window)


def test_open_world_tooltip_and_object_name_cannot_enable_hidden_controls(qtbot: object) -> None:
    """A hidden action/button cannot bypass the boundary by using metadata, not text."""
    holder = QWidget()
    qtbot.addWidget(holder)  # type: ignore[attr-defined]
    tooltip_button = QPushButton("Run", holder)
    tooltip_button.setToolTip("Unknown fruit discovery")
    registry_action = QAction("Run", holder)
    registry_action.setObjectName("future_registry_update")

    discovered = _interactive_descendants(holder)
    assert tooltip_button in discovered
    assert registry_action in discovered
    assert _advertises_open_world(tooltip_button)
    assert _advertises_open_world(registry_action)

    with pytest.raises(AssertionError, match="Open-world controls must remain disabled"):
        _assert_advertised_controls_are_disabled(holder)

    tooltip_button.setEnabled(False)
    registry_action.setEnabled(False)
    _assert_advertised_controls_are_disabled(holder)


def test_open_world_q_list_widget_item_cannot_enable_future_navigation(qtbot: object) -> None:
    """Future unknown-fruit navigation is rejected unless its item is disabled."""
    holder = QWidget()
    qtbot.addWidget(holder)  # type: ignore[attr-defined]
    navigation = QListWidget(holder)
    navigation.addItem(QListWidgetItem("Single image"))
    deferred_item = QListWidgetItem("Unknown fruit discovery")
    navigation.addItem(deferred_item)

    assert not _text_advertises_open_world(navigation.item(0).text())
    assert _text_advertises_open_world(deferred_item.text())
    with pytest.raises(AssertionError, match="Open-world controls must remain disabled"):
        _assert_advertised_controls_are_disabled(holder)

    deferred_item.setFlags(deferred_item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
    _assert_advertised_controls_are_disabled(holder)


def test_open_world_q_list_view_model_item_cannot_bypass_navigation_boundary(qtbot: object) -> None:
    """The same boundary applies when a future navigation page uses a list model."""
    holder = QWidget()
    qtbot.addWidget(holder)  # type: ignore[attr-defined]
    navigation = QListView(holder)
    model = QStandardItemModel(navigation)
    model.appendRow(QStandardItem("Batch images"))
    deferred_item = QStandardItem("Unknown fruit registry")
    model.appendRow(deferred_item)
    navigation.setModel(model)

    with pytest.raises(AssertionError, match="Open-world controls must remain disabled"):
        _assert_advertised_controls_are_disabled(holder)

    deferred_item.setEnabled(False)
    _assert_advertised_controls_are_disabled(holder)
