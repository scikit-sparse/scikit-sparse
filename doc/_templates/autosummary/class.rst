{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

{% block methods %}
{% if methods %}
.. rubric:: Methods

.. autosummary::
    :toctree:
    :nosignatures:

    {% for item in methods %}
    ~{{ name }}.{{ item }}
    {%- endfor %}
{% endif %}
{% endblock %}
