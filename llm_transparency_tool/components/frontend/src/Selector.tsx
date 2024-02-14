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
} from "streamlit-component-lib"
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';

import {
    Point,
} from './common';
import './LlmViewer.css';

export const renderParams = {
    verticalGap: 24,
    horizontalGap: 24,
    itemSize: 8,
}

interface Item {
    index: number
    text: string
    temperature: number
}

const Selector = ({ args }: ComponentProps) => {
    const items: Item[] = args["items"]
    const preselected_index: number | null = args["preselected_index"]
    const n = items.length

    const [selection, setSelection] = useState<number | null>(null)

    // Ensure the preselected element has effect only when it's a new data.
    var args_json = JSON.stringify(args)
    useEffect(() => {
        setSelection(preselected_index)
        Streamlit.setComponentValue(preselected_index)
    }, [args_json, preselected_index]);

    const handleItemClick = (index: number) => {
        setSelection(index)
        Streamlit.setComponentValue(index)
    }

    const [xScale, yScale] = useMemo(() => {
        const x = d3.scaleLinear()
            .domain([0, 1])
            .range([0, renderParams.horizontalGap])
        const y = d3.scaleLinear()
            .domain([0, n - 1])
            .range([0, renderParams.verticalGap * (n - 1)])
        return [x, y]
    }, [n])

    const itemCoords: Point[] = useMemo(() => {
        return Array.from(Array(n).keys()).map(i => ({
            x: xScale(0.5),
            y: yScale(i + 0.5),
        }))
    }, [n, xScale, yScale])

    var hasTemperature = false
    if (n > 0) {
        var t = items[0].temperature
        hasTemperature = (t !== null && t !== undefined)
    }
    const colorScale = useMemo(() => {
        var min_t = 0.0
        var max_t = 1.0
        if (hasTemperature) {
            min_t = items[0].temperature
            max_t = items[0].temperature
            for (var i = 0; i < n; i++) {
                const t = items[i].temperature
                min_t = Math.min(min_t, t)
                max_t = Math.max(max_t, t)
            }
        }
        const norm = d3.scaleLinear([min_t, max_t], [0.0, 1.0])
        const colorScale = d3.scaleSequential(d3.interpolateYlGn);
        return d3.scaleSequential(value => colorScale(norm(value)))
    }, [items, hasTemperature, n])

    const totalW = 100
    const totalH = yScale(n)
    useEffect(() => {
        Streamlit.setFrameHeight(totalH)
    }, [totalH])

    const svgRef = useRef(null);

    useEffect(() => {
        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        const getItemClass = (index: number) => {
            var style = 'selectable-item '
            style += index === selection ? 'selection' : 'selector-item'
            return style
        }

        const getItemColor = (item: Item) => {
            var t = item.temperature ?? 0.0
            return item.index === selection ? 'orange' : colorScale(t)
        }

        var icons = svg
            .selectAll('items')
            .data(Array.from(Array(n).keys()))
            .enter()
            .append('circle')
            .attr('cx', (i) => itemCoords[i].x)
            .attr('cy', (i) => itemCoords[i].y)
            .attr('r', renderParams.itemSize / 2)
            .on('click', (event: PointerEvent, i) => {
                handleItemClick(items[i].index)
            })
            .attr('class', (i) => getItemClass(items[i].index))
        if (hasTemperature) {
            icons.style('fill', (i) => getItemColor(items[i]))
        }

        svg
            .selectAll('labels')
            .data(Array.from(Array(n).keys()))
            .enter()
            .append('text')
            .attr('x', (i) => itemCoords[i].x + renderParams.horizontalGap / 2)
            .attr('y', (i) => itemCoords[i].y)
            .attr('text-anchor', 'left')
            .attr('alignment-baseline', 'middle')
            .text((i) => items[i].text)

    }, [
        items,
        n,
        itemCoords,
        selection,
        colorScale,
        hasTemperature,
    ])

    return <svg ref={svgRef} width={totalW} height={totalH}></svg>
}

export default withStreamlitConnection(Selector)
