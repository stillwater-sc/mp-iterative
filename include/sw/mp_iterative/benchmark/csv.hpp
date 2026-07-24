#pragma once
// mp-iterative benchmarking -- minimal CSV writer.
//
// Machine-readable output for plotting, alongside the human-readable summary
// tables the drivers print. Deliberately tiny and dependency-free: a header
// row plus typed row emission with consistent formatting. Fields containing a
// comma or quote are quoted per RFC 4180.

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace sw::mp_iterative::benchmark {

class csv_writer {
public:
    explicit csv_writer(std::vector<std::string> header)
        : header_(std::move(header)) {}

    void write_header(std::ostream& os) const {
        emit_row(os, header_);
    }

    /// Begin a new row; call field()/number() in column order, then end_row().
    csv_writer& field(const std::string& v) { row_.push_back(quote(v)); return *this; }

    csv_writer& number(double v) {
        std::ostringstream ss;
        ss << v;                 // default double formatting; parseable
        row_.push_back(ss.str());
        return *this;
    }

    csv_writer& number(std::size_t v) {
        row_.push_back(std::to_string(v));
        return *this;
    }

    void end_row(std::ostream& os) {
        emit_row_raw(os, row_);
        row_.clear();
    }

    std::size_t columns() const { return header_.size(); }

private:
    static std::string quote(const std::string& s) {
        if (s.find_first_of(",\"\n") == std::string::npos) return s;
        std::string out = "\"";
        for (char c : s) { if (c == '"') out += '"'; out += c; }
        out += '"';
        return out;
    }

    static void emit_row(std::ostream& os, const std::vector<std::string>& cells) {
        std::vector<std::string> quoted;
        quoted.reserve(cells.size());
        for (const auto& c : cells) quoted.push_back(quote(c));
        emit_row_raw(os, quoted);
    }

    static void emit_row_raw(std::ostream& os, const std::vector<std::string>& cells) {
        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (i) os << ',';
            os << cells[i];
        }
        os << '\n';
    }

    std::vector<std::string> header_;
    std::vector<std::string> row_;
};

} // namespace sw::mp_iterative::benchmark
