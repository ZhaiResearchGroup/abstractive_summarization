/**
 * @file pl2.cpp
 * @author CS 410 Spring 2016
 */

#include "meta/index/inverted_index.h"
#include "meta/index/ranker/pl2.h"
#include "meta/index/score_data.h"
#include "meta/math/fastapprox.h"

namespace meta
{
namespace index
{

const util::string_view pl2::id = "pl2";
const constexpr float pl2::default_c;

pl2::pl2(float c) : c_{c}
{
    // nothing
}

pl2::pl2(std::istream& in) : c_{io::packed::read<float>(in)}
{
    // nothing
}

void pl2::save(std::ostream& out) const
{
    io::packed::write(out, id);
    io::packed::write(out, c_);
}

float pl2::score_one(const score_data& sd)
{
    // TODO implement this function!

    return sd.doc_term_count; // return term frequency in doc
}

template <>
std::unique_ptr<ranker> make_ranker<pl2>(const cpptoml::table& config)
{
    auto c = config.get_as<double>("c").value_or(pl2::default_c);
    return make_unique<pl2>(c);
}
}
}
