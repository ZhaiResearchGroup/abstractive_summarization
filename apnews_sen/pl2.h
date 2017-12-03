/**
 * @file pl2.h
 * @author CS 410 Spring 2016
 *
 * All files in META are released under the MIT license. For more details,
 * consult the file LICENSE in the root of the project.
 */

#ifndef META_PL2_H_
#define META_PL2_H_

#include "meta/index/ranker/ranker.h"
#include "meta/index/ranker/ranker_factory.h"

namespace meta
{
namespace index
{

/**
 * The PL2 divergence from randomness ranking function.
 *
 * @note this is the modified version of the ranker as described in the
 * reference below
 *
 * @see Formula (4) in Hui Fang, Tao Tao, and ChengXiang Zhai. Diagnostic
 * Evaluation of Information Retrieval Models. 2011. In ACM Transactions on
 * Information Systems (TOIS), 29(2), article 7.
 *
 * Required config parameters:
 * ~~~toml
 * [ranker]
 * method = "pl2"
 * ~~~
 *
 * Optional config parameters:
 * ~~~toml
 * c = 1.0
 * ~~~
 */
class pl2 : public ranker
{
  public:
    /// Identifier for this ranker.
    const static util::string_view id;

    /// Default value of s parameter
    const static constexpr float default_c = 1.0f;

    /**
     * @param c
     */
    pl2(float c = default_c);

    /**
     * Loads a PL2 ranker from a stream.
     * @param in The stream to read from
     */
    pl2(std::istream& in);

    /**
     * @param sd the score_data for this query
     */
    float score_one(const score_data& sd) override;

    void save(std::ostream& out) const override;

  private:
    /// c parameter for PL2
    const float c_;
    static constexpr float pi_ = 3.14159265358979f;
    static constexpr float e_ = 2.718281828459045f;
};

/**
 * Specialization of the factory method used to create PL2 rankers.
 */
template <>
std::unique_ptr<ranker> make_ranker<pl2>(const cpptoml::table&);
}
}
#endif
