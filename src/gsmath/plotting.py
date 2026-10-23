"""Matplotlib-backed, label-addressable spectrum plotting."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class SpectrumPlot:
    """Pair one plotted spectrum with its Matplotlib line and style options."""
    spectrum: object
    plot_kwargs: dict = field(default_factory=dict)
    line: object = None


class SpectrumGraph:
    """Manage multiple compatible spectra on one reusable set of axes."""
    def __init__(self, *, ax=None, title=None):
        self.title = title
        self.plots = {}

        self.wavelength_unit = None
        self.value_unit = None

        self._create_axes(ax)

    def add(self, spectrum, *, label=None, replace=False, **plot_kwargs):
        """Add a labeled spectrum, optionally replacing an existing entry."""
        label = spectrum.label if label is None else label

        if label in self.plots and not replace:
            raise ValueError(f"Spectrum {label!r} is already plotted.")

        self._validate_units(spectrum)

        if label in self.plots:
            self.remove(label)

        entry = SpectrumPlot(
            spectrum=spectrum,
            plot_kwargs=dict(plot_kwargs),
        )

        self.plots[label] = entry
        self._draw_entry(label, entry)
        self._refresh()

        return self

    def remove(self, label):
        """Remove a plotted spectrum by label."""
        try:
            entry = self.plots.pop(label)
        except KeyError:
            raise KeyError(
                f"No plotted spectrum labeled {label!r}."
            ) from None

        if entry.line is not None:
            entry.line.remove()

        if not self.plots:
            self.wavelength_unit = None
            self.value_unit = None
            self._update_axis_labels()

        self._refresh()
        return self

    def update(self, spectrum, *, label=None, **plot_kwargs):
        """Replace an entry's data and merge new plotting options."""
        label = spectrum.label if label is None else label

        try:
            entry = self.plots[label]
        except KeyError:
            raise KeyError(
                f"No plotted spectrum labeled {label!r}."
            ) from None

        self._validate_units(spectrum)

        entry.spectrum = spectrum
        entry.plot_kwargs.update(plot_kwargs)

        if entry.line is not None:
            entry.line.set_data(
                spectrum.wavelength,
                spectrum.values,
            )

            if plot_kwargs:
                entry.line.set(**plot_kwargs)

        self._refresh()
        return self

    def clear(self):
        """Remove all managed spectra and redraw empty axes."""
        for entry in self.plots.values():
            if entry.line is not None:
                entry.line.remove()

        self.plots.clear()
        self.wavelength_unit = None
        self.value_unit = None

        self._update_axis_labels()
        self._refresh()

        return self

    def show(self):
        """Display the graph through Matplotlib."""
        if not plt.fignum_exists(self.figure.number):
            self._rebuild()

        self.figure.show()
        return self

    def set_yscale(self, scale):
        """Set the vertical axis scale, such as ``linear`` or ``log``."""
        self.ax.set_yscale(scale)
        self._refresh()
        return self

    def add_vline(self, x, **kwargs):
        """Add a vertical reference line."""
        self.ax.axvline(x, **kwargs)
        self._refresh()
        return self

    def add_hline(self, y, **kwargs):
        """Add a horizontal reference line."""
        self.ax.axhline(y, **kwargs)
        self._refresh()
        return self

    def add_region(self, xmin, xmax, **kwargs):
        """Shade a wavelength interval."""
        self.ax.axvspan(xmin, xmax, **kwargs)
        self._refresh()
        return self

    def _create_axes(self, ax=None):
        """Use caller axes or create a new figure/axes pair."""
        if ax is None:
            self.figure, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.figure = ax.figure

        if self.title is not None:
            self.ax.set_title(self.title)

        self._update_axis_labels()

    def _draw_entry(self, label, entry):
        """Draw one managed entry and retain its line handle."""
        line, = self.ax.plot(
            entry.spectrum.wavelength,
            entry.spectrum.values,
            label=label,
            **entry.plot_kwargs,
        )

        entry.line = line

    def _rebuild(self):
        """Redraw all entries after structural graph changes."""
        self._create_axes()

        for label, entry in self.plots.items():
            self._draw_entry(label, entry)

        self._refresh()

    def _validate_units(self, spectrum):
        """Require all plotted spectra to use compatible units."""
        if not self.plots:
            self.wavelength_unit = spectrum.wavelength_unit
            self.value_unit = spectrum.value_unit
            self._update_axis_labels()
            return

        if spectrum.wavelength_unit != self.wavelength_unit:
            raise ValueError(
                "Spectrum wavelength unit does not match graph: "
                f"{spectrum.wavelength_unit!s} != "
                f"{self.wavelength_unit!s}."
            )

        if spectrum.value_unit != self.value_unit:
            raise ValueError(
                "Spectrum value unit does not match graph: "
                f"{spectrum.value_unit!s} != "
                f"{self.value_unit!s}."
            )

    def _update_axis_labels(self):
        """Derive axis labels from the common spectrum units."""
        xlabel = "Wavelength"
        ylabel = "Value"

        if self.wavelength_unit is not None:
            xlabel += f" ({self.wavelength_unit:~})"

        if self.value_unit is not None:
            ylabel += f" ({self.value_unit:~})"

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def _refresh(self):
        """Update legend, layout, and canvas after a change."""
        self.ax.relim()
        self.ax.autoscale_view()

        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

        if self.plots:
            self.ax.legend()

        self.figure.canvas.draw_idle()
