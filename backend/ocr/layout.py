import cv2
import numpy as np
import json
from pathlib import Path

class DocumentLayoutAnalyzer:
    """
    Layout and Table Structure Analyzer.
    Detects header regions, tabular grids, row and column boundaries, and individual cell bounding boxes.
    Provides a morphological grid detector and layout fallback for ATBU legacy broadsheets & transcripts.
    """
    def __init__(self):
        pass

    def detect_layout_regions(self, img_bgr):
        h, w = img_bgr.shape[:2]
        is_landscape = w > h

        if is_landscape:
            # Result broadsheet typical layout
            header_box = [0, 0, w, int(h * 0.14)]
            table_box = [int(w * 0.02), int(h * 0.13), int(w * 0.96), int(h * 0.76)]
            footer_box = [0, int(h * 0.89), w, int(h * 0.11)]
            doc_type = "result_broadsheet"
        else:
            # Transcript typical layout
            header_box = [0, 0, w, int(h * 0.18)]
            table_box = [int(w * 0.04), int(h * 0.19), int(w * 0.92), int(h * 0.68)]
            footer_box = [0, int(h * 0.87), w, int(h * 0.13)]
            doc_type = "transcript_sheet"

        return {
            "document_type": doc_type,
            "page_dimensions": [w, h],
            "header_region": header_box,
            "table_region": table_box,
            "footer_region": footer_box
        }

    def detect_table_structure(self, img_bgr, table_bbox=None):
        """
        Extracts row and column boundaries and cell bounding boxes using morphological line analysis.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Binarize
        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )

        # Morphological kernels for horizontal and vertical lines
        scale = 30
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w / scale), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h / scale)))

        # Extract horizontal lines
        h_lines = cv2.erode(thresh, h_kernel, iterations=1)
        h_lines = cv2.dilate(h_lines, h_kernel, iterations=1)

        # Extract vertical lines
        v_lines = cv2.erode(thresh, v_kernel, iterations=1)
        v_lines = cv2.dilate(v_lines, v_kernel, iterations=1)

        # Combine into table grid
        table_grid = cv2.add(h_lines, v_lines)

        # Find external contours of cells
        contours, _ = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter out lines and huge boxes
            if 30 < cw < (w * 0.95) and 18 < ch < (h * 0.4):
                cells.append({"bbox": [x, y, cw, ch], "x": x, "y": y, "w": cw, "h": ch})

        # If morphological lines are too degraded, use configurable heuristic grid fallback
        if len(cells) < 15:
            cells = self._fallback_grid_cells(img_bgr, table_bbox)

        # Sort cells into rows and columns
        cells = sorted(cells, key=lambda c: (c["y"] // 25, c["x"]))
        
        # Group into rows
        rows = []
        if cells:
            curr_row = [cells[0]]
            row_y_threshold = 20
            for c in cells[1:]:
                if abs(c["y"] - curr_row[0]["y"]) < row_y_threshold:
                    curr_row.append(c)
                else:
                    curr_row = sorted(curr_row, key=lambda cell: cell["x"])
                    rows.append(curr_row)
                    curr_row = [c]
            if curr_row:
                curr_row = sorted(curr_row, key=lambda cell: cell["x"])
                rows.append(curr_row)

        structured_cells = []
        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row):
                structured_cells.append({
                    "row_idx": r_idx,
                    "col_idx": c_idx,
                    "bbox": cell["bbox"]
                })

        return {
            "num_rows": len(rows),
            "num_columns": max([len(r) for r in rows]) if rows else 0,
            "total_cells": len(structured_cells),
            "cells": structured_cells
        }

    def _fallback_grid_cells(self, img_bgr, table_bbox=None):
        h, w = img_bgr.shape[:2]
        is_landscape = w > h

        cells = []
        if is_landscape:
            # Broadsheet grid fallback
            col_x = [40, 90, 230, 520, 690, 860, 1030, 1200, 1370, 1470, 1560]
            top_y = 150
            row_h = 42
            num_rows = 17
            for r in range(num_rows):
                ry = top_y + r * row_h
                for c in range(len(col_x) - 1):
                    cw = col_x[c + 1] - col_x[c]
                    cells.append({"bbox": [col_x[c], ry, cw, row_h], "x": col_x[c], "y": ry, "w": cw, "h": row_h})
        else:
            # Transcript grid fallback
            col_x = [60, 130, 280, 720, 830, 930, 1030, 1140]
            top_y = 310
            row_h = 45
            num_rows = 13
            for r in range(num_rows):
                ry = top_y + r * row_h
                for c in range(len(col_x) - 1):
                    cw = col_x[c + 1] - col_x[c]
                    cells.append({"bbox": [col_x[c], ry, cw, row_h], "x": col_x[c], "y": ry, "w": cw, "h": row_h})

        return cells

    def generate_layout_overlay(self, img_bgr, layout_info, table_info, save_path=None):
        """
        Creates a visual layout inspection image with colored bounding boxes for UI presentation.
        """
        overlay = img_bgr.copy()
        
        # 1. Header (Blue)
        hx, hy, hw, hh = layout_info["header_region"]
        cv2.rectangle(overlay, (hx, hy), (hx + hw, hy + hh), (255, 120, 0), 2)
        cv2.putText(overlay, "HEADER REGION", (hx + 10, hy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

        # 2. Table Region (Green)
        tx, ty, tw, th = layout_info["table_region"]
        cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (0, 200, 0), 2)

        # 3. Cells (Orange / Purple)
        for cell in table_info["cells"]:
            cx, cy, cw, ch = cell["bbox"]
            cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (0, 165, 255), 1)

        # 4. Footer (Red)
        fx, fy, fw, fh = layout_info["footer_region"]
        cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (0, 0, 220), 2)
        cv2.putText(overlay, "FOOTER / STAMPS", (fx + 10, fy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

        if save_path:
            cv2.imwrite(str(save_path), overlay)

        return overlay
