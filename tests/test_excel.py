import xlsxwriter as xs

# Create a new workbook
wbook = xs.Workbook('/home/raspproject/yolov8counting-trackingvehicles-main/transportdata.xlsx')

# Add a worksheet
ws = wbook.add_worksheet("first_sheet")

# Write headers
ws.write(0, 0, "Date")
ws.write(0, 1, "Time stamp")
ws.write(0, 2, "confidence")
ws.write(0, 3, "x coordinate")
ws.write(0, 4, "y coordinate")
ws.write(0, 5, "pixel x1")
ws.write(0, 6, "pixel y1")
ws.write(0, 7, "pixel x2")
ws.write(0, 8, "pixel y2")

# Sample data
data = [
    ["2024-03-13", "12:34:56", 0.9, 100, 200, 50, 60, 70, 80],
    ["2024-03-14", "13:45:23", 0.8, 150, 250, 55, 65, 75, 85]
]

# Write data to the worksheet
# for row, row_data in enumerate(data, start=1):
#     for col, value in enumerate(row_data):
#         ws.write(row, col, value)

# Close the workbook
wbook.close()
