from jinja2 import Environment, FileSystemLoader
from typing import Any, Dict, List


class HtmlResultGenerator:

    html_result_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>Benchmark Results</h1>
    <table>
        <thead>
            <tr>
                {% for field in all_fields_captions %}
                <th>{{ field }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for data in sorted_data %}
            <tr>
                {% set outer_loop = loop %}
                {% for field in sort_fields[:-1] %}
                    {{ get_cell(sorted_data, outer_loop.index0, field_names, field, span_indexes) }}
                {% endfor %}
                <td>{{ data[sort_fields[-1]] }}</td>
                {% for field in result_fields %}
                    <td>{{ data['result'][field] }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

    def __init__(self):
        self.env = Environment(loader=FileSystemLoader("."))
        self.template = self.env.from_string(self.html_result_template)

    def get_cell(self, records, row_index, field_names, field, span_indexes):
        field_index = field_names.index(field)
        if row_index < len(records):
            cell_position = row_index * len(field_names) + field_index
            if span_indexes[cell_position] == 0:
                return ""
            if span_indexes[cell_position] == 1:
                return f"<td>{records[row_index][field]}</td>"
            return f'<td rowspan="{span_indexes[cell_position]}">{records[row_index][field]}</td>'
        return ""

    def generate_html(
        self,
        records: List[Dict[str, Any]],
        sort_fields: List[str],
        field_names: List[str],
        result_fields: List[str],
        all_fields_captions: List[str],
    ) -> str:
        # Sort the data by the specified fields
        sorted_data = sorted(
            records, key=lambda x: ([x[field] for field in sort_fields])
        )

        span_indexes = self.build_span_table(field_names, sorted_data)

        html = self.template.render(
            sorted_data=sorted_data,
            sort_fields=sort_fields,
            field_names=field_names,
            result_fields=result_fields,
            all_fields_captions=all_fields_captions,
            span_indexes=span_indexes,
            get_cell=self.get_cell,
        )
        return html

    # The method is calculating the span indexes for cells in an HTML table. A span index of 0 means the cell should
    # not be displayed as it will be spanned by another cell. A span index of 1 means the cell should be displayed
    # without spanning. A span index greater than 1 indicates that the  cell should span multiple rows vertically,
    # and the value represents the number of rows it should span, to reduce repetitive data and create a more
    # compact table representation.
    def build_span_table(self, field_names, records):
        # initialise array of span indexes with 1 at start for every rows in sorted_data for first X columns
        column_count = len(field_names)
        span_indexes = [1] * len(records) * column_count

        # iterate over sorted_data in reverse order
        for i in reversed(range(1, len(records))):
            first_field = field_names[0]
            if records[i - 1][first_field] == records[i][first_field]:
                first_cell_in_current_row = (i - 1) * column_count
                first_cell_in_previous_row = i * column_count
                # If the values are the same between the current row and the previous row, set the span index for the
                # current row's first cell to be equal to the span index of the previous row's first cell plus 1.
                # This means that the current row's first field should span multiple rows.
                span_indexes[first_cell_in_current_row] = (
                    span_indexes[first_cell_in_previous_row] + 1
                )
                # This means that the previous row's first field should not be displayed (since it will be spanned
                # by the current row).
                span_indexes[first_cell_in_previous_row] = 0

        for i in reversed(range(1, len(records))):
            for j, field in enumerate(field_names[1:], 1):
                cell_in_current_row = (i - 1) * column_count + j
                cell_in_previous_row = i * column_count + j
                cell_in_previous_row_and_previous_column = i * column_count + j - 1
                if (
                    records[i][field] == records[i - 1][field]
                    and span_indexes[cell_in_previous_row_and_previous_column] == 0
                ):
                    # If the values of the current field are the same between the current row and the previous row,
                    # and the span index of the previous cell in the previous row is 0 (meaning it's not being spanned),
                    # set the span index for the current cell (cell_in_current_row) to be equal to the
                    # span index of the previous cell in the same column (cell_in_previous_row) plus 1.
                    # This means that the current cell should span multiple rows.
                    span_indexes[cell_in_current_row] = (
                        span_indexes[cell_in_previous_row] + 1
                    )

                    # Set the span index for the previous cell in the same column to 0.
                    # This means that the previous cell should not be displayed (since it will be spanned by the current cell).
                    span_indexes[cell_in_previous_row] = 0

        return span_indexes

    def save_to_file(self, html: str, file_name: str) -> None:
        with open(file_name, "w") as f:
            f.write(html)
