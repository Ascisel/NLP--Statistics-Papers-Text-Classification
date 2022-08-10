import csv
import src.file_names as file_names
import os
import glob
from xlsxwriter.workbook import Workbook
import csv



def prepere_xls_reports() -> None:
    """
    Concat csv files into one big xlsx file where each sheet corresponds to each csv file.
    """
    classification_results = [
        file_names.best_results_dir,
        file_names.best_results_dir,
        file_names.uber_results_dir,
    ]
    report_files = [
        file_names.best_report_summary,
        file_names.best_report_summary,
        file_names.report_uber_summary,
    ]
    for results, file in zip(classification_results, report_files):
        workbook = Workbook(file)
        for csvfile in glob.glob(os.path.join(results, "*.csv")):
            sheet_name = csvfile.replace(results + "/", "")
            worksheet = workbook.add_worksheet(sheet_name[-7:])
            with open(csvfile, "rt", encoding="utf8") as f:
                reader = csv.reader(f)
                for r, row in enumerate(reader):
                    for c, col in enumerate(row):
                        worksheet.write(r, c, col)
        workbook.close()
