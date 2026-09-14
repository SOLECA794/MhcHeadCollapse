
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MHC_HEAD_COLLAPSE_H_
#define ACLNN_MHC_HEAD_COLLAPSE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMhcHeadCollapseGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * hcBase : required
 * hcScale : required
 * epsNorm : optional
 * epsHc : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMhcHeadCollapseGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *hcBase,
    const aclTensor *hcScale,
    double epsNorm,
    double epsHc,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMhcHeadCollapse
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMhcHeadCollapse(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
