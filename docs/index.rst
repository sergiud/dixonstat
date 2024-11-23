dixonstat
=========

``dixonstat`` implements numerical evaluation of critical values for Dixon's
:math:`r_{j,i}` statistics :cite:`Dixon1950` following the methods introduced by
:cite:t:`McBane2006` and :cite:t:`Verma2014`.

Given a set of :math:`n` observations :math:`x_i` ordered such that
:math:`x_1\leq x_2\leq \dotsb \leq x_n`, the statistics are defined by

.. math::

    r_{j,i-1} = \frac{x_n-x_{n-j}}{x_n-x_i}

Dixon's :math:`r_{1,0}` statistic (i.e., :math:`i=j=1`) is often called
:math:`Q` and the corresponding outlier rejection test which uses this ratio is
called the :math:`Q` test.

The ratio :math:`r_{1,0}`, for instance, simply compares the difference between
a single suspected outlier (:math:`x_1` or :math:`x_n`) and its
nearest-neighboring value to the overall range of values in the sample. In other
words, the ratio determines the fraction of the total range that is attributable
to one suspected outlier.

Different numerical approaches exist for generating the critical values of
Dixon's :math:`r` statistics. A straightforward method is to interpolate new
confidence levels using cubic regression from previously tabulated data as
suggested by :cite:t:`Rorabacher1991`. Given critical values were originally
tabulated for relatively small sample sizes (i.e., :math:`3\leq n\leq 30`),
interpolation might not always be feasible.

Without interpolation, determining critical values boils down to integrating the
probability density function to obtain the cumulative distribution function. The
integration can be performed either using a stochastic approach
:cite:`Efstathiou1992` (e.g., by means of Monte Carlo simulation) or using
Gaussian quadrature. The latter numerical evaluation of the corresponding
integral is employed by ``dixonstat``.

.. toctree::
   :hidden:

   usage
   bibliography
   license
