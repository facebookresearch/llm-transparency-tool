/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {
    ComponentProps,
    Streamlit,
    withStreamlitConnection,
} from 'streamlit-component-lib'
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';

import {
    Label,
    Point,
} from './common';
import './LlmViewer.css';

export const renderParams = {
    cellH: 32,
    cellW: 32,
    attnSize: 8,
    afterFfnSize: 8,
    ffnSize: 6,
    tokenSelectorSize: 16,
    layerCornerRadius: 6,
}

interface Cell {
    layer: number
    token: number
}

enum CellItem {
    AfterAttn = 'after_attn',
    AfterFfn = 'after_ffn',
    Ffn = 'ffn',
    Original = 'original',  // They will only be at level = 0
}

interface Node {
    cell: Cell | null
    item: CellItem | null
}

interface NodeProps {
    node: Node
    pos: Point
    isActive: boolean
}

interface EdgeRaw {
    weight: number
    source: string
    target: string
}

interface Edge {
    weight: number
    from: Node
    to: Node
    fromPos: Point
    toPos: Point
    isSelectable: boolean
    isFfn: boolean
}

interface Selection {
    node: Node | null
    edge: Edge | null
}

function tokenPointerPolygon(origin: Point) {
    const r = renderParams.tokenSelectorSize / 2
    const dy = r / 2
    const dx = r * Math.sqrt(3.0) / 2
    // Draw an arrow looking down
    return [
        [origin.x, origin.y + r],
        [origin.x + dx, origin.y - dy],
        [origin.x - dx, origin.y - dy],
    ].toString()
}

function isSameCell(cell1: Cell | null, cell2: Cell | null) {
    if (cell1 == null || cell2 == null) {
        return false
    }
    return cell1.layer === cell2.layer && cell1.token === cell2.token
}

function isSameNode(node1: Node | null, node2: Node | null) {
    if (node1 === null || node2 === null) {
        return false
    }
    return isSameCell(node1.cell, node2.cell)
        && node1.item === node2.item;
}

function isSameEdge(edge1: Edge | null, edge2: Edge | null) {
    if (edge1 === null || edge2 === null) {
        return false
    }
    return isSameNode(edge1.from, edge2.from) && isSameNode(edge1.to, edge2.to);
}

function nodeFromString(name: string) {
    const match = name.match(/([AIMX])(\d+)_(\d+)/)
    if (match == null) {
        return {
            cell: null,
            item: null,
        }
    }
    const [, type, layerStr, tokenStr] = match
    const layer = +layerStr
    const token = +tokenStr

    const typeToCellItem = new Map<string, CellItem>([
        ['A', CellItem.AfterAttn],
        ['I', CellItem.AfterFfn],
        ['M', CellItem.Ffn],
        ['X', CellItem.Original],
    ])
    return {
        cell: {
            layer: layer,
            token: token,
        },
        item: typeToCellItem.get(type) ?? null,
    }
}

function isValidNode(node: Node, nLayers: number, nTokens: number) {
    if (node.cell === null) {
        return true
    }
    return node.cell.layer < nLayers && node.cell.token < nTokens
}

function isValidSelection(selection: Selection, nLayers: number, nTokens: number) {
    if (selection.node !== null) {
        return isValidNode(selection.node, nLayers, nTokens)
    }
    if (selection.edge !== null) {
        return isValidNode(selection.edge.from, nLayers, nTokens) &&
            isValidNode(selection.edge.to, nLayers, nTokens)
    }
    return true
}

const ContributionGraph = ({ args }: ComponentProps) => {
    const modelInfo = args['model_info']
    const tokens = args['tokens']
    const edgesRaw: EdgeRaw[][] = args['edges_per_token']

    const nLayers = modelInfo === null ? 0 : modelInfo.n_layers
    const nTokens = tokens === null ? 0 : tokens.length

    const [selection, setSelection] = useState<Selection>({
        node: null,
        edge: null,
    })
    var curSelection = selection
    if (!isValidSelection(selection, nLayers, nTokens)) {
        curSelection = {
            node: null,
            edge: null,
        }
        setSelection(curSelection)
        Streamlit.setComponentValue(curSelection)
    }

    const [startToken, setStartToken] = useState<number>(nTokens - 1)
    // We have startToken state var, but it won't be updated till next render, so use
    // this var in the current render.
    var curStartToken = startToken
    if (startToken >= nTokens) {
        curStartToken = nTokens - 1
        setStartToken(curStartToken)
    }

    const handleRepresentationClick = (node: Node) => {
        const newSelection: Selection = {
            node: node,
            edge: null,
        }
        setSelection(newSelection)
        Streamlit.setComponentValue(newSelection)
    }

    const handleEdgeClick = (edge: Edge) => {
        if (!edge.isSelectable) {
            return
        }
        const newSelection: Selection = {
            node: edge.to,
            edge: edge,
        }
        setSelection(newSelection)
        Streamlit.setComponentValue(newSelection)
    }

    const handleTokenClick = (t: number) => {
        setStartToken(t)
    }

    const [xScale, yScale] = useMemo(() => {
        const x = d3.scaleLinear()
            .domain([-2, nTokens - 1])
            .range([0, renderParams.cellW * (nTokens + 2)])
        const y = d3.scaleLinear()
            .domain([-1, nLayers])
            .range([renderParams.cellH * (nLayers + 2), 0])
        return [x, y]
    }, [nLayers, nTokens])

    const cells = useMemo(() => {
        let result: Cell[] = []
        for (let l = 0; l < nLayers; l++) {
            for (let t = 0; t < nTokens; t++) {
                result.push({
                    layer: l,
                    token: t,
                })
            }
        }
        return result
    }, [nLayers, nTokens])

    const nodeCoords = useMemo(() => {
        let result = new Map<string, Point>()
        const w = renderParams.cellW
        const h = renderParams.cellH
        for (var cell of cells) {
            const cx = xScale(cell.token + 0.5)
            const cy = yScale(cell.layer - 0.5)
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.AfterAttn }),
                { x: cx, y: cy + h / 4 },
            )
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.AfterFfn }),
                { x: cx, y: cy - h / 4 },
            )
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.Ffn }),
                { x: cx + 5 * w / 16, y: cy },
            )
        }
        for (let t = 0; t < nTokens; t++) {
            cell = {
                layer: 0,
                token: t,
            }
            const cx = xScale(cell.token + 0.5)
            const cy = yScale(cell.layer - 1.0)
            result.set(
                JSON.stringify({ cell: cell, item: CellItem.Original }),
                { x: cx, y: cy + h / 4 },
            )
        }
        return result
    }, [cells, nTokens, xScale, yScale])

    const edges: Edge[][] = useMemo(() => {
        let result = []
        for (var edgeList of edgesRaw) {
            let edgesPerStartToken = []
            for (var edge of edgeList) {
                const u = nodeFromString(edge.source)
                const v = nodeFromString(edge.target)
                var isSelectable = (
                    u.cell !== null && v.cell !== null && v.item === CellItem.AfterAttn
                )
                var isFfn = (
                    u.cell !== null && v.cell !== null && (
                        u.item === CellItem.Ffn || v.item === CellItem.Ffn
                    )
                )
                edgesPerStartToken.push({
                    weight: edge.weight,
                    from: u,
                    to: v,
                    fromPos: nodeCoords.get(JSON.stringify(u)) ?? { 'x': 0, 'y': 0 },
                    toPos: nodeCoords.get(JSON.stringify(v)) ?? { 'x': 0, 'y': 0 },
                    isSelectable: isSelectable,
                    isFfn: isFfn,
                })
            }
            result.push(edgesPerStartToken)
        }
        return result
    }, [edgesRaw, nodeCoords])

    const activeNodes = useMemo(() => {
        let result = new Set<string>()
        for (var edge of edges[curStartToken]) {
            const u = JSON.stringify(edge.from)
            const v = JSON.stringify(edge.to)
            result.add(u)
            result.add(v)
        }
        return result
    }, [edges, curStartToken])

    const nodeProps = useMemo(() => {
        let result: Array<NodeProps> = []
        nodeCoords.forEach((p: Point, node: string) => {
            result.push({
                node: JSON.parse(node),
                pos: p,
                isActive: activeNodes.has(node),
            })
        })
        return result
    }, [nodeCoords, activeNodes])

    const tokenLabels: Label[] = useMemo(() => {
        if (!tokens) {
            return []
        }
        return tokens.map((s: string, i: number) => ({
            text: s.replace(/ /g, '·'),
            pos: {
                x: xScale(i + 0.5),
                y: yScale(-1.5),
            },
        }))
    }, [tokens, xScale, yScale])

    const layerLabels: Label[] = useMemo(() => {
        return Array.from(Array(nLayers).keys()).map(i => ({
            text: 'L' + i,
            pos: {
                x: xScale(-0.25),
                y: yScale(i - 0.5),
            },
        }))
    }, [nLayers, xScale, yScale])

    const tokenSelectors: Array<[number, Point]> = useMemo(() => {
        return Array.from(Array(nTokens).keys()).map(i => ([
            i,
            {
                x: xScale(i + 0.5),
                y: yScale(nLayers - 0.5),
            }
        ]))
    }, [nTokens, nLayers, xScale, yScale])

    const totalW = xScale(nTokens + 2)
    const totalH = yScale(-4)
    useEffect(() => {
        Streamlit.setFrameHeight(totalH)
    }, [totalH])

    const colorScale = d3.scaleLinear(
        [0.0, 0.5, 1.0],
        ['#9eba66', 'darkolivegreen', 'darkolivegreen']
    )
    const ffnEdgeColorScale = d3.scaleLinear(
        [0.0, 0.5, 1.0],
        ['orchid', 'purple', 'purple']
    )
    const edgeWidthScale = d3.scaleLinear([0.0, 0.5, 1.0], [2.0, 3.0, 3.0])

    const svgRef = useRef(null);

    useEffect(() => {
        const getNodeStyle = (p: NodeProps, type: string) => {
            if (isSameNode(p.node, curSelection.node)) {
                return 'selectable-item selection'
            }
            if (p.isActive) {
                return 'selectable-item active-' + type + '-node'
            }
            return 'selectable-item inactive-node'
        }

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        svg
            .selectAll('layers')
            .data(Array.from(Array(nLayers).keys()).filter((x) => x % 2 === 1))
            .enter()
            .append('rect')
            .attr('class', 'layer-highlight')
            .attr('x', xScale(-1.0))
            .attr('y', (layer) => yScale(layer))
            .attr('width', xScale(nTokens + 0.25) - xScale(-1.0))
            .attr('height', (layer) => yScale(layer) - yScale(layer + 1))
            .attr('rx', renderParams.layerCornerRadius)

        svg
            .selectAll('edges')
            .data(edges[curStartToken])
            .enter()
            .append('line')
            .style('stroke', (edge: Edge) => {
                if (isSameEdge(edge, curSelection.edge)) {
                    return 'orange'
                }
                if (edge.isFfn) {
                    return ffnEdgeColorScale(edge.weight)
                }
                return colorScale(edge.weight)
            })
            .attr('class', (edge: Edge) => edge.isSelectable ? 'selectable-edge' : '')
            .style('stroke-width', (edge: Edge) => edgeWidthScale(edge.weight))
            .attr('x1', (edge: Edge) => edge.fromPos.x)
            .attr('y1', (edge: Edge) => edge.fromPos.y)
            .attr('x2', (edge: Edge) => edge.toPos.x)
            .attr('y2', (edge: Edge) => edge.toPos.y)
            .on('click', (event: PointerEvent, edge) => {
                handleEdgeClick(edge)
            })

        svg
            .selectAll('residual')
            .data(nodeProps)
            .enter()
            .filter((p) => {
                return p.node.item === CellItem.AfterAttn
                    || p.node.item === CellItem.AfterFfn
            })
            .append('circle')
            .attr('class', (p) => getNodeStyle(p, 'residual'))
            .attr('cx', (p) => p.pos.x)
            .attr('cy', (p) => p.pos.y)
            .attr('r', renderParams.attnSize / 2)
            .on('click', (event: PointerEvent, p) => {
                handleRepresentationClick(p.node)
            })

        svg
            .selectAll('ffn')
            .data(nodeProps)
            .enter()
            .filter((p) => p.node.item === CellItem.Ffn && p.isActive)
            .append('rect')
            .attr('class', (p) => getNodeStyle(p, 'ffn'))
            .attr('x', (p) => p.pos.x - renderParams.ffnSize / 2)
            .attr('y', (p) => p.pos.y - renderParams.ffnSize / 2)
            .attr('width', renderParams.ffnSize)
            .attr('height', renderParams.ffnSize)
            .on('click', (event: PointerEvent, p) => {
                handleRepresentationClick(p.node)
            })

        svg
            .selectAll('token_labels')
            .data(tokenLabels)
            .enter()
            .append('text')
            .attr('x', (label: Label) => label.pos.x)
            .attr('y', (label: Label) => label.pos.y)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('alignment-baseline', 'top')
            .attr('transform', (label: Label) =>
                'rotate(-40, ' + label.pos.x + ', ' + label.pos.y + ')')
            .text((label: Label) => label.text)

        svg
            .selectAll('layer_labels')
            .data(layerLabels)
            .enter()
            .append('text')
            .attr('x', (label: Label) => label.pos.x)
            .attr('y', (label: Label) => label.pos.y)
            .attr('text-anchor', 'middle')
            .attr('alignment-baseline', 'middle')
            .text((label: Label) => label.text)

        svg
            .selectAll('token_selectors')
            .data(tokenSelectors)
            .enter()
            .append('polygon')
            .attr('class', ([i,]) => (
                curStartToken === i
                    ? 'selectable-item selection'
                    : 'selectable-item token-selector'
            ))
            .attr('points', ([, p]) => tokenPointerPolygon(p))
            .attr('r', renderParams.tokenSelectorSize / 2)
            .on('click', (event: PointerEvent, [i,]) => {
                handleTokenClick(i)
            })
    }, [
        cells,
        edges,
        nodeProps,
        tokenLabels,
        layerLabels,
        tokenSelectors,
        curStartToken,
        curSelection,
        colorScale,
        ffnEdgeColorScale,
        edgeWidthScale,
        nLayers,
        nTokens,
        xScale,
        yScale
    ])

    return <svg ref={svgRef} width={totalW} height={totalH}></svg>
}

export default withStreamlitConnection(ContributionGraph)
