#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>

#include "nms_trt.h"

struct Point2D {
    float x, y;
};

static std::vector<Point2D> get_rotated_corners(const Box2d& box) {
    float cx = (box.x1 + box.x2) / 2.0f;
    float cy = (box.y1 + box.y2) / 2.0f;
    float dx = box.x2 - box.x1;
    float dy = box.y2 - box.y1;
    float h = box.theta;
    
    float cos_h = std::cos(h);
    float sin_h = std::sin(h);
    
    float dx2 = dx / 2.0f;
    float dy2 = dy / 2.0f;
    
    std::vector<Point2D> corners(4);
    
    // Corners in clockwise order
    corners[0].x = dx2 * cos_h - dy2 * sin_h + cx;
    corners[0].y = dx2 * sin_h + dy2 * cos_h + cy;
    
    corners[1].x = dx2 * cos_h - (-dy2) * sin_h + cx;
    corners[1].y = dx2 * sin_h + (-dy2) * cos_h + cy;
    
    corners[2].x = (-dx2) * cos_h - (-dy2) * sin_h + cx;
    corners[2].y = (-dx2) * sin_h + (-dy2) * cos_h + cy;
    
    corners[3].x = (-dx2) * cos_h - dy2 * sin_h + cx;
    corners[3].y = (-dx2) * sin_h + dy2 * cos_h + cy;
    
    return corners;
}

static std::vector<Point2D> clip_by_edge(const std::vector<Point2D>& poly, const Point2D& a, const Point2D& b) {
    std::vector<Point2D> out;
    int n = poly.size();
    if (n == 0) {
        return out;
    }
    for (int i = 0; i < n; ++i) {
        Point2D c = poly[i];
        Point2D nx = poly[(i + 1) % n];
        
        float dc = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        float dn = (b.x - a.x) * (nx.y - a.y) - (b.y - a.y) * (nx.x - a.x);
        
        if (dc <= 0.0f) {
            out.push_back(c);
        }
        if ((dc <= 0.0f) != (dn <= 0.0f)) {
            float denom = dc - dn;
            if (std::abs(denom) > 1e-8f) {
                float t = dc / denom;
                Point2D intersect;
                intersect.x = c.x + t * (nx.x - c.x);
                intersect.y = c.y + t * (nx.y - c.y);
                out.push_back(intersect);
            }
        }
    }
    return out;
}

static float polygon_intersection_area(const std::vector<Point2D>& poly1, const std::vector<Point2D>& poly2) {
    if (poly1.empty() || poly2.empty()) return 0.0f;

    // Fast AABB early-exit
    float min_x1 = poly1[0].x, max_x1 = poly1[0].x;
    float min_y1 = poly1[0].y, max_y1 = poly1[0].y;
    for (const auto& p : poly1) {
        min_x1 = std::min(min_x1, p.x);
        max_x1 = std::max(max_x1, p.x);
        min_y1 = std::min(min_y1, p.y);
        max_y1 = std::max(max_y1, p.y);
    }
    float min_x2 = poly2[0].x, max_x2 = poly2[0].x;
    float min_y2 = poly2[0].y, max_y2 = poly2[0].y;
    for (const auto& p : poly2) {
        min_x2 = std::min(min_x2, p.x);
        max_x2 = std::max(max_x2, p.x);
        min_y2 = std::min(min_y2, p.y);
        max_y2 = std::max(max_y2, p.y);
    }

    if (max_x1 < min_x2 || min_x1 > max_x2 || max_y1 < min_y2 || min_y1 > max_y2) {
        return 0.0f;
    }

    std::vector<Point2D> clipped = poly1;
    int n2 = poly2.size();
    for (int i = 0; i < n2; ++i) {
        clipped = clip_by_edge(clipped, poly2[i], poly2[(i + 1) % n2]);
        if (clipped.empty()) {
            return 0.0f;
        }
    }

    if (clipped.size() < 3) {
        return 0.0f;
    }

    // Shoelace formula
    float area = 0.0f;
    int num_pts = clipped.size();
    for (int i = 0; i < num_pts; ++i) {
        int next_i = (i + 1) % num_pts;
        area += clipped[i].x * clipped[next_i].y - clipped[next_i].x * clipped[i].y;
    }
    return 0.5f * std::abs(area);
}

float iou(const Box2d& a, const Box2d& b) {
    auto corners_a = get_rotated_corners(a);
    auto corners_b = get_rotated_corners(b);
    
    float interArea = polygon_intersection_area(corners_a, corners_b);
    
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    
    float unionArea = area_a + area_b - interArea;
    if (unionArea < 1e-8f) {
        return 0.0f;
    }
    return interArea / unionArea;
}

void nms(std::vector<Box2d>& bboxes_2d_filtered, std::vector<float>& scores, 
         float& nms_thr, std::vector<int>& nms_filter_inds) {
    std::vector<int> indices(bboxes_2d_filtered.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });

    while (!indices.empty()) {
        int idx = indices.front();
        nms_filter_inds.push_back(idx);
        
        indices.erase(indices.begin());
        
        for (auto it = indices.begin(); it != indices.end(); ) {
            if (iou(bboxes_2d_filtered[idx], bboxes_2d_filtered[*it]) > nms_thr) {
                it = indices.erase(it);
            } else {
                ++it;
            }
        }
    }
}